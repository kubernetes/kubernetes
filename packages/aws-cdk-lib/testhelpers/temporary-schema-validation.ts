/**
 * Helpers for loading temporary CFN schemas into the validation plugin during tests.
 *
 * During simultaneous releases, pre-GA CloudFormation properties live in
 * tools/@aws-cdk/spec2cdk/temporary-schemas/. This module discovers those schemas
 * and provides the directory path for configuring the validation plugin singleton.
 *
 * Used by jest-global-app-testhook.ts — not intended for production runtime.
 */

import * as fs from 'fs';
import * as path from 'path';
import { UnscopedValidationError } from '../core/lib/errors';
import { lit } from '../core/lib/private/literal-string';

/**
 * Resolve the spec2cdk temporary-schemas directory if it exists and contains schemas.
 * In the public repo this directory only has a .keep file (returns undefined).
 * In aws-cdk-private it contains pre-GA CFN schemas (returns the path).
 */
function findTemporarySchemasDirectory(): string | undefined {
  const candidate = path.resolve(__dirname, '../../..', 'tools/@aws-cdk/spec2cdk/temporary-schemas');
  if (!fs.existsSync(candidate)) {
    return undefined;
  }
  return hasSchemaFiles(candidate) ? candidate : undefined;
}

/**
 * Recursively check if a directory contains any .json files.
 * Skips symlinks for safety.
 */
function hasSchemaFiles(dir: string): boolean {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.isSymbolicLink()) continue;
    if (entry.isFile() && entry.name.endsWith('.json')) return true;
    if (entry.isDirectory() && hasSchemaFiles(path.join(dir, entry.name))) return true;
  }
  return false;
}

// Computed once at module load — cached for the lifetime of the test process.
const TEMPORARY_SCHEMAS_DIR = findTemporarySchemasDirectory();

/**
 * Whether temporary schemas are present in this repo checkout.
 * True in aws-cdk-private with pre-GA schemas, false in the public repo.
 */
export function hasTemporarySchemas(): boolean {
  return TEMPORARY_SCHEMAS_DIR !== undefined;
}

/**
 * Get the temporary schemas directory path.
 * @throws if no temporary schemas are present
 */
export function getTemporarySchemasDirectory(): string {
  if (!TEMPORARY_SCHEMAS_DIR) {
    throw new UnscopedValidationError(lit`NoTemporarySchemas`, 'No temporary schemas directory found');
  }
  return TEMPORARY_SCHEMAS_DIR;
}
