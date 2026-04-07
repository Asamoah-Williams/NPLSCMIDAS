from npl_src.db import DatabaseConnection


EXCLUDED_PROCEDURES = {
    "test_FailProcedure",
    "test_RecalculateData",
    "test_RunMonthlyUpdate",
    "usp_Liquidity_LLM_Series",
    "usp_Calc_EVE_DurationGap",
    "usp_IR_InterestRateVaR_1d10d",
    "usp_IRRBB_NII_EaR_BucketedExposureWithNet",
    "usp_IRRBB_NII_EaR_FundingSourceBucket",
    "usp_IRRBB_NII_EaR_OverallEaRAnnualised",
    "usp_IRRBB_EVE_DurationGap_ModDur_GHS",
    "usp_IRRBB_EVE_DurationGap_Oct2025",
    "usp_IRRBB_EVE_DurationGap_Single",
    "usp_IRRBB_NII_EaR_12m"

}


def load_allowed_procedures():
    query = """
        SELECT 
            p.name AS ProcedureName,
            SCHEMA_NAME(p.schema_id) AS SchemaName
        FROM sys.procedures p
        ORDER BY p.name;
    """

    conn = DatabaseConnection()

    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    allowed = {}
    idx = 1

    for proc_name, schema_name in rows:
        if proc_name in EXCLUDED_PROCEDURES:
            continue

        allowed[idx] = {
            "name": proc_name,
            "description": f"Executes procedure {proc_name}",
            "db_proc": f"{schema_name}.{proc_name}"
        }

        idx += 1

    return allowed


def list_procedures():
    allowed = load_allowed_procedures()

    return [
        {
            "id": pid,
            "name": p["name"],
            "description": p["description"]
        }
        for pid, p in allowed.items()
    ]


def execute_procedures(procedure_ids):
    results = []
    allowed = load_allowed_procedures()

    conn = DatabaseConnection()

    cursor = conn.cursor()

    for pid in procedure_ids:
        proc = allowed.get(pid)

        if not proc:
            results.append({
                "id": pid,
                "status": "SKIPPED",
                "reason": "Not allowlisted"
            })
            continue

        try:
            cursor.execute(f"EXEC {proc['db_proc']}")
            conn.commit()

            results.append({
                "id": pid,
                "procedure": proc["name"],
                "status": "SUCCESS"
            })

        except Exception as e:
            conn.rollback()

            results.append({
                "id": pid,
                "procedure": proc["name"],
                "status": "FAILED",
                "error": str(e)
            })

    cursor.close()
    conn.close()

    return {
        "executed_count": len(results),
        "results": results
    }
