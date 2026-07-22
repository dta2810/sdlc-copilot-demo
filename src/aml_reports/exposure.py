"""Counterparty exposure reporting for AML review.

This module implements the **high‑risk exposure report** required by the
Jira ticket ``SAN‑2264``.  The report aggregates the ``kyc_screening`` table
by ``screening_result`` and joins the ``party`` and ``counterparty`` reference
tables to enrich the output with country information.  Parties of type ``PEP``
are flagged as high‑priority.  Personally‑identifiable columns such as
``tax_id_number`` and ``national_id_number`` are deliberately omitted from the
result.

The implementation is deliberately defensive – column names are referenced
explicitly but the function does not assume the presence of any additional
columns.  If a required column is missing the Spark job will raise an error,
making the issue visible during testing.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, countDistinct, lit, when


def high_risk_exposure_report(spark: SparkSession) -> DataFrame:
	"""Generate a high‑risk exposure report.

	The function performs the following steps:
	1. Load the three source tables from the Unity Catalog schema
	   ``main_david_thomas.customer``.
	2. Select only the columns required for the report, explicitly excluding
	   the PII columns ``tax_id_number`` and ``national_id_number``.
	3. Join ``kyc_screening`` with ``party`` (on ``party_id``) and with
	   ``counterparty`` (on ``counterparty_id``) to obtain ``country_of_residence``
	   and ``country_code``.
	4. Add a ``high_priority`` flag that is ``True`` when ``party_type`` equals
	   ``"PEP"``.
	5. Group by ``screening_result`` together with the country fields and the
	   priority flag, counting distinct parties and counterparties.

	Parameters
	----------
	spark: SparkSession
		An active Spark session.

	Returns
	-------
	DataFrame
		A DataFrame with the columns:
		``screening_result``, ``country_of_residence``, ``country_code``,
		``high_priority``, ``counterparty_count`` and ``party_count``.
	"""

	# ---------------------------------------------------------------------
	# 1. Load source tables – the fully‑qualified name includes catalog and
	#    schema.  Adjust the catalog name if your workspace uses a different
	#    one.
	# ---------------------------------------------------------------------
	kyc_tbl = "main_david_thomas.customer.kyc_screening"
	party_tbl = "main_david_thomas.customer.party"
	counterparty_tbl = "main_david_thomas.customer.counterparty"

	kyc_df = spark.table(kyc_tbl)
	party_df = spark.table(party_tbl)
	counterparty_df = spark.table(counterparty_tbl)

	# ---------------------------------------------------------------------
	# 2. Select only the columns we need.  We deliberately omit any PII
	#    columns that may exist in the source tables.
	# ---------------------------------------------------------------------
	party_sel = party_df.select(
		col("party_id"),
		col("party_type"),  # used to flag PEPs
		col("country_of_residence"),
		col("country_code"),
	)

	counterparty_sel = counterparty_df.select(
		col("counterparty_id"),
		col("country_of_residence"),
		col("country_code"),
	)

	# ---------------------------------------------------------------------
	# 3. Join the tables.  Left joins preserve rows from ``kyc_screening`` even
	#    if a reference record is missing.
	# ---------------------------------------------------------------------
	joined = (
		kyc_df.join(party_sel, on="party_id", how="left")
		.join(counterparty_sel, on="counterparty_id", how="left")
	)

	# ---------------------------------------------------------------------
	# 4. Flag high‑priority parties (PEP).
	# ---------------------------------------------------------------------
	flagged = joined.withColumn(
		"high_priority", when(col("party_type") == "PEP", lit(True)).otherwise(lit(False))
	)

	# ---------------------------------------------------------------------
	# 5. Aggregate the results.
	# ---------------------------------------------------------------------
	result = (
		flagged.groupBy(
			"screening_result",
			"country_of_residence",
			"country_code",
			"high_priority",
		)
		.agg(
			countDistinct("counterparty_id").alias("counterparty_count"),
			countDistinct("party_id").alias("party_count"),
		)
	)

	return result
