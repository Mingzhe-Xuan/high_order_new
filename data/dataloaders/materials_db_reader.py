import sqlite3
import numpy as np
import json
from typing import List, Dict, Any, Optional
from ase import Atoms

class MaterialsProjectDatabase:
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
    def _parse_numbers_blob(blob: Optional[bytes]) -> Optional[np.ndarray]:
        if blob is None:
            return None
        return np.frombuffer(blob, dtype=np.int32)
    
    @staticmethod
    def _parse_positions_blob(blob: Optional[bytes]) -> Optional[np.ndarray]:
        if blob is None:
            return None
        return np.frombuffer(blob, dtype=np.float64).reshape(-1, 3)
    
    @staticmethod
    def _parse_cell_blob(blob: Optional[bytes]) -> Optional[np.ndarray]:
        if blob is None:
            return None
        return np.frombuffer(blob, dtype=np.float64).reshape(3, 3)
    
    @staticmethod
    def _parse_blob(blob: Optional[bytes]) -> Optional[np.ndarray]:
        if blob is None:
            return None
        return np.frombuffer(blob, dtype=np.float64)
    
    def get_row_by_id(self, row_id: int) -> Optional[Dict[str, Any]]:
        self.cursor.execute('SELECT * FROM systems WHERE id = ?', (row_id,))
        row = self.cursor.fetchone()
        if row is None:
            return None
        return self._parse_row(row)
    
    def get_all_rows(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if limit:
            self.cursor.execute('SELECT * FROM systems LIMIT ?', (limit,))
        else:
            self.cursor.execute('SELECT * FROM systems')
        rows = self.cursor.fetchall()
        return [self._parse_row(row) for row in rows]
    
    def get_atoms_by_id(self, row_id: int) -> Optional[Atoms]:
        data = self.get_row_by_id(row_id)
        if data is None:
            return None
        return self._create_atoms(data)
    
    def get_all_atoms(self, limit: Optional[int] = None) -> List[Atoms]:
        rows = self.get_all_rows(limit)
        return [self._create_atoms(data) for data in rows if data is not None]
    
    def _parse_row(self, row: tuple) -> Dict[str, Any]:
        columns = ['id', 'unique_id', 'ctime', 'mtime', 'username', 
                   'numbers', 'positions', 'cell', 'pbc', 
                   'initial_magmoms', 'initial_charges', 'masses', 
                   'tags', 'momenta', 'constraints', 'calculator', 
                   'calculator_parameters', 'energy', 'free_energy', 
                   'forces', 'stress', 'dipole', 'magmoms', 
                   'magmom', 'charges', 'key_value_pairs', 'data', 
                   'natoms', 'fmax', 'smax', 'volume', 'mass', 'charge']
        
        data_dict = {}
        for col_name, value in zip(columns, row):
            if col_name == 'numbers':
                data_dict[col_name] = self._parse_numbers_blob(value)
            elif col_name == 'positions':
                data_dict[col_name] = self._parse_positions_blob(value)
            elif col_name == 'cell':
                data_dict[col_name] = self._parse_cell_blob(value)
            elif col_name in ['initial_magmoms', 'initial_charges', 'masses', 
                               'tags', 'momenta', 'forces', 'stress', 
                               'dipole', 'magmoms', 'charges']:
                data_dict[col_name] = self._parse_blob(value)
            elif col_name == 'key_value_pairs':
                data_dict[col_name] = json.loads(value) if value else None
            else:
                data_dict[col_name] = value
        
        return data_dict
    
    def _create_atoms(self, data: Dict[str, Any]) -> Atoms:
        atoms = Atoms(
            numbers=data['numbers'],
            positions=data['positions'],
            cell=data['cell'],
            pbc=bool(data['pbc'])
        )
        return atoms
    
    def get_material_ids(self, limit: Optional[int] = None) -> List[str]:
        if limit:
            self.cursor.execute('SELECT key_value_pairs FROM systems LIMIT ?', (limit,))
        else:
            self.cursor.execute('SELECT key_value_pairs FROM systems')
        rows = self.cursor.fetchall()
        material_ids = []
        for row in rows:
            if row[0]:
                try:
                    kv = json.loads(row[0])
                    if 'material_id' in kv:
                        material_ids.append(kv['material_id'])
                except:
                    pass
        return material_ids
    
    def get_row_count(self) -> int:
        self.cursor.execute('SELECT COUNT(*) FROM systems')
        return self.cursor.fetchone()[0]


if __name__ == "__main__":
    db_path = r'e:\课业\课题组\high_order\data\materials_project_full.db'
    
    with MaterialsProjectDatabase(db_path) as db:
        print(f"总共有 {db.get_row_count()} 条记录")
        
        print("\n" + "="*80)
        print("获取第一条记录的 Atoms 对象:")
        print("="*80)
        
        atoms = db.get_atoms_by_id(1)
        if atoms:
            print(f"原子数: {len(atoms)}")
            print(f"原子序数: {atoms.get_atomic_numbers()}")
            print(f"原子位置:\n{atoms.get_positions()}")
            print(f"晶胞:\n{atoms.get_cell()}")
            print(f"体积: {atoms.get_volume()}")
        
        print("\n" + "="*80)
        print("获取前 5 条记录的原始数据:")
        print("="*80)
        
        rows = db.get_all_rows(limit=5)
        for i, row in enumerate(rows):
            print(f"\n记录 {i+1}:")
            print(f"  ID: {row['id']}")
            print(f"  Unique ID: {row['unique_id']}")
            print(f"  原子数: {row['natoms']}")
            print(f"  原子序数: {row['numbers']}")
            print(f"  体积: {row['volume']}")
            if row['key_value_pairs']:
                print(f"  Material ID: {row['key_value_pairs'].get('material_id')}")
