import Database from "better-sqlite3";
import path from "path";

const DB_PATH = path.resolve(
  process.cwd(),
  "../Processed_Data.db"
);

let _db: Database.Database | null = null;

export function getDb(): Database.Database {
  if (!_db) {
    _db = new Database(DB_PATH, { readonly: true, fileMustExist: true });
  }
  return _db;
}
