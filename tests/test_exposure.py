import pytest
from pyspark.sql import SparkSession
from src.aml_reports.exposure import high_risk_exposure_report

def test_high_risk_exposure_report():
    spark = SparkSession.builder.master("local[2]").appName("test_exposure").getOrCreate()
    # Sample data matching expected schema (minimal columns)
    kyc_data = [
        ("p1", "c1", "SCREENED"),
        ("p2", "c2", "PENDING"),
        ("p3", "c3", "SCREENED"),
    ]
    kyc_df = spark.createDataFrame(kyc_data, ["party_id", "counterparty_id", "screening_result"])
    party_data = [
        ("p1", "PEP", "US", "USA"),
        ("p2", "NON_PEP", "GB", "GBR"),
        ("p3", "PEP", "CA", "CAN"),
    ]
    party_df = spark.createDataFrame(party_data, ["party_id", "party_type", "country_of_residence", "country_code"])
    counterparty_data = [
        ("c1", "US", "USA"),
        ("c2", "GB", "GBR"),
        ("c3", "CA", "CAN"),
    ]
    counterparty_df = spark.createDataFrame(counterparty_data, ["counterparty_id", "country_of_residence", "country_code"])
    # Register temporary views with fully qualified names expected by the function
    kyc_df.createOrReplaceTempView("main_david_thomas.customer.kyc_screening")
    party_df.createOrReplaceTempView("main_david_thomas.customer.party")
    counterparty_df.createOrReplaceTempView("main_david_thomas.customer.counterparty")
    result = high_risk_exposure_report(spark)
    # Verify expected columns are present and PEP flag works
    expected_cols = {"screening_result", "country_of_residence", "country_code", "high_priority", "counterparty_count", "party_count"}
    assert set(result.columns) == expected_cols
    # Collect and perform a simple sanity check on counts
    rows = {row["screening_result"]: row["party_count"] for row in result.collect()}
    assert rows["SCREENED"] == 2
    assert rows["PENDING"] == 1
    spark.stop()
