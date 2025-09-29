"""Persistence layer for vector database with versioning and recovery."""

import json
import pickle
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import shutil


class PersistenceManager:
    """Manager for persisting vector indices with versioning and backup."""

    def __init__(
        self,
        base_path: str,
        format: str = "pickle",
        enable_versioning: bool = True,
        max_versions: int = 10
    ):
        """Initialize persistence manager.

        Args:
            base_path: Base directory for storage
            format: Storage format ('pickle', 'hdf5', 'custom')
            enable_versioning: Whether to keep version history
            max_versions: Maximum number of versions to keep
        """
        self.base_path = Path(base_path)
        self.format = format
        self.enable_versioning = enable_versioning
        self.max_versions = max_versions

        self.base_path.mkdir(parents=True, exist_ok=True)

        # Version tracking
        self.versions: List[str] = []
        self._load_versions()

    def _load_versions(self) -> None:
        """Load version history."""
        versions_file = self.base_path / "versions.json"
        if versions_file.exists():
            with open(versions_file, 'r') as f:
                self.versions = json.load(f)

    def _save_versions(self) -> None:
        """Save version history."""
        versions_file = self.base_path / "versions.json"
        with open(versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)

    def _generate_version_id(self) -> str:
        """Generate unique version ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}"

    def save_pickle(self, data: Any, name: str, version: Optional[str] = None) -> str:
        """Save data using pickle format.

        Args:
            data: Data to save
            name: Name for the data
            version: Optional version ID

        Returns:
            Version ID
        """
        if version is None and self.enable_versioning:
            version = self._generate_version_id()

        # Create version directory
        if version:
            save_dir = self.base_path / version
        else:
            save_dir = self.base_path

        save_dir.mkdir(parents=True, exist_ok=True)

        # Save data
        file_path = save_dir / f"{name}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Update versions
        if version and version not in self.versions:
            self.versions.append(version)
            self._save_versions()
            self._cleanup_old_versions()

        return version or "latest"

    def load_pickle(self, name: str, version: Optional[str] = None) -> Any:
        """Load data from pickle format.

        Args:
            name: Name of the data
            version: Optional version ID (defaults to latest)

        Returns:
            Loaded data
        """
        if version:
            load_dir = self.base_path / version
        else:
            # Load from latest version
            if self.versions:
                load_dir = self.base_path / self.versions[-1]
            else:
                load_dir = self.base_path

        file_path = load_dir / f"{name}.pkl"

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_hdf5(
        self,
        vectors: np.ndarray,
        metadata: Dict,
        name: str,
        version: Optional[str] = None
    ) -> str:
        """Save vectors and metadata in HDF5 format.

        Args:
            vectors: Vector array (N, dim)
            metadata: Metadata dictionary
            name: Name for the data
            version: Optional version ID

        Returns:
            Version ID
        """
        if version is None and self.enable_versioning:
            version = self._generate_version_id()

        # Create version directory
        if version:
            save_dir = self.base_path / version
        else:
            save_dir = self.base_path

        save_dir.mkdir(parents=True, exist_ok=True)

        # Save to HDF5
        file_path = save_dir / f"{name}.h5"

        with h5py.File(file_path, 'w') as f:
            # Save vectors
            f.create_dataset('vectors', data=vectors, compression='gzip')

            # Save metadata as JSON string
            f.attrs['metadata'] = json.dumps(metadata)

        # Update versions
        if version and version not in self.versions:
            self.versions.append(version)
            self._save_versions()
            self._cleanup_old_versions()

        return version or "latest"

    def load_hdf5(self, name: str, version: Optional[str] = None) -> tuple:
        """Load vectors and metadata from HDF5 format.

        Args:
            name: Name of the data
            version: Optional version ID

        Returns:
            Tuple of (vectors, metadata)
        """
        if version:
            load_dir = self.base_path / version
        else:
            if self.versions:
                load_dir = self.base_path / self.versions[-1]
            else:
                load_dir = self.base_path

        file_path = load_dir / f"{name}.h5"

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with h5py.File(file_path, 'r') as f:
            vectors = f['vectors'][:]
            metadata = json.loads(f.attrs['metadata'])

        return vectors, metadata

    def save(self, data: Any, name: str, version: Optional[str] = None) -> str:
        """Save data using configured format.

        Args:
            data: Data to save
            name: Name for the data
            version: Optional version ID

        Returns:
            Version ID
        """
        if self.format == "pickle":
            return self.save_pickle(data, name, version)
        elif self.format == "hdf5":
            if isinstance(data, tuple) and len(data) == 2:
                vectors, metadata = data
                return self.save_hdf5(vectors, metadata, name, version)
            else:
                raise ValueError("HDF5 format requires (vectors, metadata) tuple")
        else:
            raise ValueError(f"Unknown format: {self.format}")

    def load(self, name: str, version: Optional[str] = None) -> Any:
        """Load data using configured format.

        Args:
            name: Name of the data
            version: Optional version ID

        Returns:
            Loaded data
        """
        if self.format == "pickle":
            return self.load_pickle(name, version)
        elif self.format == "hdf5":
            return self.load_hdf5(name, version)
        else:
            raise ValueError(f"Unknown format: {self.format}")

    def _cleanup_old_versions(self) -> None:
        """Remove old versions if exceeding max_versions."""
        while len(self.versions) > self.max_versions:
            old_version = self.versions.pop(0)
            old_dir = self.base_path / old_version

            if old_dir.exists():
                shutil.rmtree(old_dir)

        self._save_versions()

    def list_versions(self) -> List[str]:
        """List all available versions.

        Returns:
            List of version IDs
        """
        return self.versions.copy()

    def delete_version(self, version: str) -> bool:
        """Delete a specific version.

        Args:
            version: Version ID to delete

        Returns:
            True if deleted
        """
        if version not in self.versions:
            return False

        # Remove directory
        version_dir = self.base_path / version
        if version_dir.exists():
            shutil.rmtree(version_dir)

        # Remove from versions list
        self.versions.remove(version)
        self._save_versions()

        return True

    def create_snapshot(self, name: str) -> str:
        """Create a named snapshot of current state.

        Args:
            name: Snapshot name

        Returns:
            Version ID
        """
        version = f"snapshot_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Copy latest version
        if self.versions:
            latest_dir = self.base_path / self.versions[-1]
            snapshot_dir = self.base_path / version

            shutil.copytree(latest_dir, snapshot_dir)

            self.versions.append(version)
            self._save_versions()

        return version

    def restore_snapshot(self, version: str) -> bool:
        """Restore from a specific snapshot.

        Args:
            version: Version ID to restore from

        Returns:
            True if restored
        """
        if version not in self.versions:
            return False

        # Create new version from snapshot
        new_version = self._generate_version_id()
        source_dir = self.base_path / version
        dest_dir = self.base_path / new_version

        shutil.copytree(source_dir, dest_dir)

        self.versions.append(new_version)
        self._save_versions()

        return True

    def get_size(self, version: Optional[str] = None) -> Dict[str, int]:
        """Get storage size information.

        Args:
            version: Optional version ID

        Returns:
            Dictionary with size information in bytes
        """
        if version:
            target_dir = self.base_path / version
        else:
            target_dir = self.base_path

        if not target_dir.exists():
            return {'total_bytes': 0, 'num_files': 0}

        total_size = 0
        num_files = 0

        for path in target_dir.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
                num_files += 1

        return {
            'total_bytes': total_size,
            'total_mb': total_size / (1024 ** 2),
            'total_gb': total_size / (1024 ** 3),
            'num_files': num_files
        }

    def export_metadata(self, version: Optional[str] = None) -> Dict:
        """Export metadata about stored data.

        Args:
            version: Optional version ID

        Returns:
            Metadata dictionary
        """
        metadata = {
            'base_path': str(self.base_path),
            'format': self.format,
            'enable_versioning': self.enable_versioning,
            'max_versions': self.max_versions,
            'versions': self.versions,
            'current_version': self.versions[-1] if self.versions else None,
            'storage_size': self.get_size(version)
        }

        return metadata