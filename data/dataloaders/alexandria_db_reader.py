import sqlite3
import numpy as np
import json
from typing import List, Dict, Any, Optional
from ase import Atoms
import pickle as pkl

class AlexandriaDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
    
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    @staticmethod
    def _parse_blob(blob: Optional[bytes]) -> Optional[np.ndarray]:
        if blob is None:
            return None
        return np.frombuffer(blob, dtype=np.float64)
    
    def get_row_by_id(self, row_id: int) -> Optional[Dict[str, Any]]:
        self.cursor.execute('SELECT * FROM data WHERE idx = ?', (row_id,))
        row = self.cursor.fetchone()
        if row is None:
            return None
        return self._parse_row(row)
    
    def get_structure_by_id(self, row_id: int) -> Optional[Dict[str, Any]]:
        data = self.get_row_by_id(row_id)
        if data is None:
            return None
        # Use pickle to decode byte formatted pymatgen structure
        return pkl.loads(data["structure"])
    
    def get_energy_by_id(self, row_id: int) -> Optional[float]:
        data = self.get_row_by_id(row_id)
        if data is None:
            return None
        return data["energy"]
    
    def get_all_rows(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if limit:
            self.cursor.execute('SELECT * FROM data LIMIT ?', (limit,))
        else:
            self.cursor.execute('SELECT * FROM data')
        rows = self.cursor.fetchall()
        return [self._parse_row(row) for row in rows]
    
    def _parse_row(self, row: tuple) -> Dict[str, Any]:
        columns = ['structure', 'energy']
        
        data_dict = {}
        for col_name, value in zip(columns, row):
            data_dict[col_name] = value
        
        return data_dict
    
    def get_row_count(self) -> int:
        self.cursor.execute('SELECT COUNT(*) FROM data')
        return self.cursor.fetchone()[0]
