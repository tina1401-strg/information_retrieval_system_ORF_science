import sqlite3
from pathlib import Path
import time
from article import Article
from file_utilities import LAST_UPDATE_FILE


class DBHandler:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_table()
        #self.updated_ids = self.update()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id          INTEGER PRIMARY KEY,
                url         TEXT,
                title       TEXT,
                date        TEXT,
                description TEXT,
                markdown    TEXT,
                image_url   TEXT
            )
        """)
        self.conn.commit()
    
    def needs_update(self) -> bool:
        """Returns True if last update was more than 10 hours ago."""
        path = Path(LAST_UPDATE_FILE)
        if not path.exists():
            return True
        last = float(path.read_text())
        if(time.time() - last > 2592000):
            print("\033[33mWarning: Database has not been updated in over 30 days. Some recent articles may be missing.\033[0m")
        return (time.time() - last) > 10 * 3600

    def mark_updated(self):
        """Save current timestamp as last update time."""
        path = Path(LAST_UPDATE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(time.time()))

    async def update(self):
        if not self.needs_update():
            print("DB is up to date, skipping update.")
            self.updated_ids = []
            return
        """Fetch new articles from ORF and insert them into the database."""
        # imported here to avoid circular imports
        af = Article()
        existing_ids = self.get_existing_ids()          # BUG FIX: removed wrong extra `self` argument
        new_ids = af.scrape_story_ids()
        new_ids = [i for i in new_ids if i not in existing_ids]
        print(f"Fetching {len(new_ids)} new articles...")
        new_articles = await af.fetch_stories(new_ids) 
        inserted = self.insert_articles(new_articles)
        self.updated_ids = [a["id"] for a in new_articles]
        self.mark_updated()
        print(f"Inserted {inserted} new articles.")

    def insert_articles(self, articles: list) -> int:
        """Insert a list of article dicts, skipping duplicates. Returns count inserted."""
        inserted = 0
        for article in articles:
            cursor = self.conn.execute("""
                INSERT OR IGNORE INTO articles (id, url, title, date, description, markdown, image_url)
                VALUES (:id, :url, :title, :date, :description, :markdown, :image_url)
            """, {
                "id":          article.get("id"),
                "url":         article.get("url"),
                "title":       article.get("title"),
                "date":        article.get("date"),
                "description": article.get("description"),
                "markdown":    article.get("markdown"),
                "image_url":   article.get("image_url"),
            })
            inserted += cursor.rowcount
        self.conn.commit()
        return inserted

    def get_existing_ids(self) -> set:
        """Return all article IDs already in the database."""
        rows = self.conn.execute("SELECT id FROM articles").fetchall()
        return {row["id"] for row in rows}

    def get_all_articles(self) -> list:
        """Return all articles as list of dicts — used for building indexes."""
        rows = self.conn.execute(
            "SELECT id, url, title, date, description, markdown, image_url FROM articles"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_article_by_id(self, article_id: int) -> dict | None:
        """Return a single article by ID — used after search to return result."""
        row = self.conn.execute(
            "SELECT * FROM articles WHERE id = ?", (article_id,)
        ).fetchone()
        return dict(row) if row else None

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
