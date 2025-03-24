import Database from 'better-sqlite3'
import path from 'path'

// Get the database path from environment variable or use default
const dbPath = process.env.DATABASE_PATH 
  ? path.resolve(process.cwd(), process.env.DATABASE_PATH)
  : path.resolve(process.cwd(), '../data/ufc_fighters.db')

export function getDb() {
  return new Database(dbPath, { readonly: true })
}

// Export an empty object to make this a module
export {} 