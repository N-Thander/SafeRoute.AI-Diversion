import sqlite3

# Connect to the SQLite database
db_file = "tracking_info.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Fetch all rows from the 'tracking' table
cursor.execute("SELECT * FROM tracking")
rows = cursor.fetchall()

# Print the column names
columns = [description[0] for description in cursor.description]
print(columns)

# Print the data
for row in rows:
    print(row)

# Close the connection
conn.close()
