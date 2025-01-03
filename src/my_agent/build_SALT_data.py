import sqlite3
import pandas as pd
from tqdm import tqdm

# -------------------------------------------------------------------
# 1. Configure file paths and table definitions
# -------------------------------------------------------------------
# Parquet files:
salesdocuments_parquet = "parquet_files/I_SalesDocument_train.parquet"
salesdoc_items_parquet = "parquet_files/I_SalesDocumentItem_train.parquet"
customers_parquet = "parquet_files/I_Customer.parquet"
addresses_parquet = "parquet_files/I_AddrOrgNamePostalAddress.parquet"

# SQLite database output:
sqlite_db_file = "salt_data.sqlite"

# Chunk size for batch inserts:
CHUNK_SIZE = 10_000

# Columns for each table (in the same order we will insert them).
# Make sure these match exactly your Parquet column names.
sales_doc_columns = [
    "SALESDOCUMENT",
    "SALESOFFICE",
    "SALESGROUP",
    "CUSTOMERPAYMENTTERMS",
    "SHIPPINGCONDITION",
    "SALESDOCUMENTTYPE",
    "SALESORGANIZATION",
    "DISTRIBUTIONCHANNEL",
    "ORGANIZATIONDIVISION",
    "BILLINGCOMPANYCODE",
    "TRANSACTIONCURRENCY",
    "INCOTERMSCLASSIFICATION",
    "CREATIONDATE",
    "CREATIONTIME"
]

sales_item_columns = [
    "SALESDOCUMENT",
    "SALESDOCUMENTITEM",
    "PLANT",
    "SHIPPINGPOINT",
    "SALESDOCUMENTITEMCATEGORY",
    "PRODUCT",
    "SOLDTOPARTY",
    "SHIPTOPARTY",
    "BILLTOPARTY",
    "PAYERPARTY",
    "INCOTERMSCLASSIFICATION"
]

customer_columns = [
    "CUSTOMER",
    "ADDRESSID"
]

address_columns = [
    "ADDRESSID",
    "ADDRESSREPRESENTATIONCODE",
    "COUNTRY",
    "REGION"
]

# -------------------------------------------------------------------
# 2. Create an empty SQLite database with the table definitions
# -------------------------------------------------------------------
conn = sqlite3.connect(sqlite_db_file)
# Enable foreign key enforcement
conn.execute("PRAGMA foreign_keys = ON;")

# Drop tables if you want a fresh run each time (optional).
# Comment these out if you want to append data in an existing DB.
conn.execute("DROP TABLE IF EXISTS I_SalesDocumentItem;")
conn.execute("DROP TABLE IF EXISTS I_SalesDocument;")
conn.execute("DROP TABLE IF EXISTS I_Customer;")
conn.execute("DROP TABLE IF EXISTS I_AddrOrgNamePostalAddress;")

# Create the tables with primary/foreign keys and TEXT columns
# (change TEXT to INTEGER, DATE, etc. if appropriate).
conn.execute("""
CREATE TABLE I_SalesDocument (
    SALESDOCUMENT             TEXT PRIMARY KEY,
    SALESOFFICE               TEXT,
    SALESGROUP                TEXT,
    CUSTOMERPAYMENTTERMS      TEXT,
    SHIPPINGCONDITION         TEXT,
    SALESDOCUMENTTYPE         TEXT,
    SALESORGANIZATION         TEXT,
    DISTRIBUTIONCHANNEL       TEXT,
    ORGANIZATIONDIVISION      TEXT,
    BILLINGCOMPANYCODE        TEXT,
    TRANSACTIONCURRENCY       TEXT,
    INCOTERMSCLASSIFICATION   TEXT,
    CREATIONDATE              TEXT,
    CREATIONTIME              TEXT
);
""")

conn.execute("""
CREATE TABLE I_SalesDocumentItem (
    SALESDOCUMENT               TEXT,
    SALESDOCUMENTITEM           TEXT,
    PLANT                       TEXT,
    SHIPPINGPOINT               TEXT,
    SALESDOCUMENTITEMCATEGORY   TEXT,
    PRODUCT                     TEXT,
    SOLDTOPARTY                 TEXT,
    SHIPTOPARTY                 TEXT,
    BILLTOPARTY                 TEXT,
    PAYERPARTY                  TEXT,
    INCOTERMSCLASSIFICATION     TEXT,
    PRIMARY KEY (SALESDOCUMENT, SALESDOCUMENTITEM),
    FOREIGN KEY (SALESDOCUMENT) REFERENCES I_SalesDocument(SALESDOCUMENT),
    FOREIGN KEY (SOLDTOPARTY) REFERENCES I_Customer(CUSTOMER),
    FOREIGN KEY (SHIPTOPARTY) REFERENCES I_Customer(CUSTOMER),
    FOREIGN KEY (BILLTOPARTY) REFERENCES I_Customer(CUSTOMER),
    FOREIGN KEY (PAYERPARTY) REFERENCES I_Customer(CUSTOMER)
);
""")

conn.execute("""
CREATE TABLE I_Customer (
    CUSTOMER   TEXT PRIMARY KEY,
    ADDRESSID  TEXT,
    FOREIGN KEY (ADDRESSID) REFERENCES I_AddrOrgNamePostalAddress(ADDRESSID)
);
""")

conn.execute("""
CREATE TABLE I_AddrOrgNamePostalAddress (
    ADDRESSID                  TEXT PRIMARY KEY,
    ADDRESSREPRESENTATIONCODE  TEXT,
    COUNTRY                    TEXT,
    REGION                     TEXT
);
""")

conn.commit()

# -------------------------------------------------------------------
# 3. Define helper functions for chunking + inserting
# -------------------------------------------------------------------
def insert_dataframe_in_chunks(df, insert_statement, db_connection, chunk_size=10000):
    """
    Insert rows from a DataFrame into SQLite in batches to improve performance.
    """
    # Convert DataFrame to list of tuples (row-wise).
    # This can still be large if df is huge, so you might want to chunk
    # at the DataFrame level, too. We'll do a simple approach here.
    all_rows = df.itertuples(index=False, name=None)
    
    buffer = []
    count_inserted = 0

    db_connection.execute("BEGIN TRANSACTION;")
    try:
        for row in tqdm(all_rows):
            buffer.append(row)
            if len(buffer) == chunk_size:
                db_connection.executemany(insert_statement, buffer)
                buffer.clear()
                count_inserted += chunk_size

        # Insert any leftover rows
        if buffer:
            db_connection.executemany(insert_statement, buffer)
            count_inserted += len(buffer)
            buffer.clear()
        db_connection.execute("COMMIT;")
    except Exception:
        db_connection.execute("ROLLBACK;")
        raise

    return count_inserted

# -------------------------------------------------------------------
# 4. Read each Parquet file with pandas in chunk-friendly ways
#    and insert the data into each table
# -------------------------------------------------------------------
#
# Note: If the dataset is extremely large, consider reading Parquet
# in smaller chunks using PyArrow + partial scans. For moderate size,
# reading into a single DataFrame is typically okay.
#
# We'll show the simpler approach (read the entire file once) below.
# For truly massive data, see PyArrow's iter_batches approach or
# chunked iteration in Pandas.




# -----------------------
# I_AddrOrgNamePostalAddress
# -----------------------
print(f"Reading {addresses_parquet}...")
df_addresses = pd.read_parquet(addresses_parquet, columns=address_columns)

insert_stmt = f"""
INSERT OR IGNORE INTO I_AddrOrgNamePostalAddress
({", ".join(address_columns)})
VALUES ({", ".join(["?" for _ in address_columns])})
"""
num_inserted = insert_dataframe_in_chunks(
    df_addresses,
    insert_stmt,
    conn,
    chunk_size=CHUNK_SIZE
)
print(f"Inserted {num_inserted} rows into I_AddrOrgNamePostalAddress.")

# -----------------------
# I_Customer
# -----------------------
print(f"Reading {customers_parquet}...")
df_customers = pd.read_parquet(customers_parquet, columns=customer_columns)

insert_stmt = f"""
INSERT OR IGNORE INTO I_Customer
({", ".join(customer_columns)})
VALUES ({", ".join(["?" for _ in customer_columns])})
"""
num_inserted = insert_dataframe_in_chunks(
    df_customers,
    insert_stmt,
    conn,
    chunk_size=CHUNK_SIZE
)
print(f"Inserted {num_inserted} rows into I_Customer.")


# -----------------------
# I_SalesDocument
# -----------------------
print(f"Reading {salesdocuments_parquet}...")
df_salesdocuments = pd.read_parquet(salesdocuments_parquet, columns=sales_doc_columns)
# Example: Convert to string before inserting into SQLite
df_salesdocuments["CREATIONDATE"] = df_salesdocuments["CREATIONDATE"].astype(str)
df_salesdocuments["CREATIONTIME"] = df_salesdocuments["CREATIONTIME"].astype(str)


insert_stmt = f"""
INSERT OR IGNORE INTO I_SalesDocument
({", ".join(sales_doc_columns)})
VALUES ({", ".join(["?" for _ in sales_doc_columns])})
"""
num_inserted = insert_dataframe_in_chunks(
    df_salesdocuments,
    insert_stmt,
    conn,
    chunk_size=CHUNK_SIZE
)
print(f"Inserted {num_inserted} rows into I_SalesDocument.")

# -----------------------
# I_SalesDocumentItem
# -----------------------
print(f"Reading {salesdoc_items_parquet}...")
df_salesdocument_items = pd.read_parquet(salesdoc_items_parquet, columns=sales_item_columns)

insert_stmt = f"""
INSERT OR IGNORE INTO I_SalesDocumentItem
({", ".join(sales_item_columns)})
VALUES ({", ".join(["?" for _ in sales_item_columns])})
"""
num_inserted = insert_dataframe_in_chunks(
    df_salesdocument_items,
    insert_stmt,
    conn,
    chunk_size=CHUNK_SIZE
)
print(f"Inserted {num_inserted} rows into I_SalesDocumentItem.")





# -------------------------------------------------------------------
# 5. Finalize / close
# -------------------------------------------------------------------
conn.close()
print("Done! The SALT tables have been saved into", sqlite_db_file)
