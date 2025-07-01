from database.config import db_manager
db_manager.connect()

# Check existing tables
tables = db_manager.execute_query("""
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'diffusion'
""")
print('Existing tables:')
print(tables)

# Check selfies table columns
try:
    selfies_cols = db_manager.execute_query("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_schema = 'diffusion' AND table_name = 'selfies'
    """)
    print('\nSelfies table columns:')
    print(selfies_cols)
except:
    print('\nSelfies table does not exist')

# Check frames table columns  
try:
    frames_cols = db_manager.execute_query("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_schema = 'diffusion' AND table_name = 'frames'
    """)
    print('\nFrames table columns:')
    print(frames_cols)
except:
    print('\nFrames table does not exist')
