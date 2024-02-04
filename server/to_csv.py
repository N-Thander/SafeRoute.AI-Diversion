import sqlite3
import csv

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

# Specify the CSV file path
csv_file_path = "tracking_info.csv"

# Write data to the CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write header
    csv_writer.writerow(columns)
    
    # Write rows
    csv_writer.writerows(rows)

print(f"Data exported to {csv_file_path}")

# Close the connection
conn.close()
