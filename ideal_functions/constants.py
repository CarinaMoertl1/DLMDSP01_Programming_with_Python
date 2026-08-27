"""Shared constants used across the package.

Kept in one place so every module rounds x-values to the same
precision. Any mismatch here would let the mapping table produced
by ``analysis.py`` silently drift out of sync with the x-lookups
performed in ``visualization.py``.
"""

# Round x values before using them as lookup keys.
# This prevents small floating-point differences between the
# training, ideal, and test CSV data from causing lookup errors.
X_DECIMALS = 9