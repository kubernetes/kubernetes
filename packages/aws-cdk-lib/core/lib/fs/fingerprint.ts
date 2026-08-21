import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { FingerprintDiskCache } from './fingerprint-disk-cache';
import { IgnoreStrategy } from './ignore';
import type { FingerprintOptions } from './options';
import { IgnoreMode, SymlinkFollowMode } from './options';
import { isInternalPath, resolveLinkTarget } from './utils';
import { UnscopedValidationError } from '../errors';
import { lit } from '../private/literal-string';

const CTRL_SOH = '\x01';
const CTRL_SOT = '\x02';
const CTRL_ETX = '\x03';

/**
 * Files are fingerprinted only the first time they are encountered, to save
 * time hashing large files. This function clears this cache, should it be
 * necessary for some reason.
 */
export function clearLargeFileFingerprintCache() {
  // No-op: caches are now per-operation and scoped to the fingerprint() call.
  // Retained for API compatibility.
}

/**
 * Produces fingerprint based on the contents of a single file or an entire directory tree.
 *
 * Line endings are converted from CRLF to LF.
 *
 * The fingerprint will also include:
 * 1. An extra string if defined in `options.extra`.
 * 2. The symlink follow mode value.
 *
 * @param fileOrDirectory The directory or file to fingerprint
 * @param options Fingerprinting options
 */
export function fingerprint(fileOrDirectory: string, options: FingerprintOptions = { }) {
  const hash = crypto.createHash('sha256');
  _hashField(hash, 'options.extra', options.extraHash || '');
  const follow = options.follow || SymlinkFollowMode.EXTERNAL;
  _hashField(hash, 'options.follow', follow);

  // Resolve symlinks in the initial path (for example, the root directory
  // might be symlinked). It's important that we know the absolute path, so we
  // can judge if further symlinks inside the target directory are within the
  // target or not (if we don't resolve, we would test w.r.t. the wrong path).
  fileOrDirectory = fs.realpathSync(fileOrDirectory);

  const isDir = fs.statSync(fileOrDirectory).isDirectory();
  const rootDirectory = isDir
    ? fileOrDirectory
    : path.dirname(fileOrDirectory);

  const ignoreMode = options.ignoreMode || IgnoreMode.GLOB;
  if (ignoreMode != IgnoreMode.GLOB) {
    _hashField(hash, 'options.ignoreMode', ignoreMode);
  }

  // Pre-resolve rootDirectory once — avoids repeated path.resolve in the hot loop
  const resolvedRoot = path.resolve(rootDirectory);

  const ignoreStrategy = IgnoreStrategy.fromCopyOptions(options, fileOrDirectory);

  // Per-operation disk cache scoped to this directory
  const cache = new FingerprintDiskCache(resolvedRoot);

  function _contentFingerprint(file: string): string {
    const stats = fs.statSync(file, { bigint: true });
    return contentFingerprintWithStats(file, stats, cache);
  }

  // Dispatch based on whether the root is a file or directory
  if (isDir) {
    _processDirectory(fileOrDirectory, fileOrDirectory);
  } else {
    const hashComponent = path.relative(fileOrDirectory, fileOrDirectory).replace(/\\/g, '/');
    _hashField(hash, `file:${hashComponent}`, _contentFingerprint(fileOrDirectory));
  }

  cache.save();
  return hash.digest('hex');

  // --- Inlined shouldFollow logic (avoids per-call path.resolve + fs.existsSync overhead) ---

  function _shouldFollowLink(resolvedLinkTarget: string): boolean {
    switch (follow) {
      case SymlinkFollowMode.ALWAYS:
        return true;
      case SymlinkFollowMode.EXTERNAL:
        return !isInternalPath(resolvedRoot, resolvedLinkTarget);
      case SymlinkFollowMode.BLOCK_EXTERNAL:
        return isInternalPath(resolvedRoot, resolvedLinkTarget);
      case SymlinkFollowMode.NEVER:
        return false;
      default:
        return false;
    }
  }

  // --- Core traversal ---

  function _processDirectory(symbolicPath: string, realPath: string) {
    const entries = fs.readdirSync(realPath, { withFileTypes: true });
    const sorted = entries.sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
    for (const entry of sorted) {
      const childSymbolicPath = path.join(symbolicPath, entry.name);
      const childRealPath = path.join(realPath, entry.name);
      if (entry.isSymbolicLink()) {
        _processSymlink(childSymbolicPath, childRealPath);
      } else if (entry.isFile()) {
        if (ignoreStrategy.ignores(childSymbolicPath)) {
          continue;
        }
        const hashComponent = path.relative(fileOrDirectory, childSymbolicPath).replace(/\\/g, '/');
        _hashField(hash, `file:${hashComponent}`, _contentFingerprint(childRealPath));
      } else if (entry.isDirectory()) {
        if (ignoreStrategy.completelyIgnores(childSymbolicPath)) {
          continue;
        }
        _processDirectory(childSymbolicPath, childRealPath);
      }
    }
  }

  function _processSymlink(symbolicPath: string, realPath: string) {
    const linkTarget = fs.readlinkSync(realPath);
    const resolvedLinkTarget = resolveLinkTarget(realPath, linkTarget);

    if (!_shouldFollowLink(resolvedLinkTarget)) {
      // Not following — hash the link target string itself
      if (ignoreStrategy.ignores(symbolicPath)) {
        return;
      }
      const hashComponent = path.relative(fileOrDirectory, symbolicPath).replace(/\\/g, '/');
      _hashField(hash, `link:${hashComponent}`, linkTarget);
      return;
    }

    // Following the symlink — stat the target to determine type
    let targetStat: fs.BigIntStats;
    try {
      targetStat = fs.statSync(resolvedLinkTarget, { bigint: true });
    } catch {
      // Target doesn't exist — treat as non-followed link
      if (ignoreStrategy.ignores(symbolicPath)) {
        return;
      }
      const hashComponent = path.relative(fileOrDirectory, symbolicPath).replace(/\\/g, '/');
      _hashField(hash, `link:${hashComponent}`, linkTarget);
      return;
    }

    if (targetStat.isDirectory()) {
      if (ignoreStrategy.completelyIgnores(symbolicPath)) {
        return;
      }
      _processDirectory(symbolicPath, resolvedLinkTarget);
    } else if (targetStat.isFile()) {
      if (ignoreStrategy.ignores(symbolicPath)) {
        return;
      }
      const hashComponent = path.relative(fileOrDirectory, symbolicPath).replace(/\\/g, '/');
      _hashField(hash, `file:${hashComponent}`, contentFingerprintWithStats(resolvedLinkTarget, targetStat, cache));
    } else {
      throw new UnscopedValidationError(
        lit`UnableToUnableHashNeither`,
        `Unable to hash ${symbolicPath}: it is neither a file nor a directory`,
      );
    }
  }
}

export function contentFingerprint(file: string): string {
  const stats = fs.statSync(file, { bigint: true });
  return contentFingerprintWithStats(file, stats, undefined);
}

function contentFingerprintWithStats(file: string, stats: fs.BigIntStats, cache: FingerprintDiskCache | undefined): string {
  const cacheKey = `${stats.ino}|${stats.mtimeMs}|${stats.size}`;
  return cache?.obtain(cacheKey, () => contentFingerprintMiss(file)) ?? contentFingerprintMiss(file);
}

// Pre-compiled regex for CRLF normalization
const CRLF_RE = /\r\n/g;
const TRAILING_CR_RE = /\r$/;
const BUFFER_SIZE = 8 * 1024;

function contentFingerprintMiss(file: string): string {
  const hash = crypto.createHash('sha256');
  const buffer = Buffer.alloc(BUFFER_SIZE);
  const fd = fs.openSync(file, fs.constants.O_RDONLY);
  let size = 0;
  let isBinary = false;
  let lastStr = '';
  let read = 0;
  try {
    while ((read = fs.readSync(fd, buffer, 0, BUFFER_SIZE, null)) !== 0) {
      const slicedBuffer = buffer.subarray(0, read);

      if (size === 0) {
        isBinary = slicedBuffer.indexOf(0) !== -1;
      }

      if (isBinary) {
        size += read;
        hash.update(slicedBuffer);
      } else {
        const str = slicedBuffer.toString();

        if (TRAILING_CR_RE.test(str)) {
          lastStr += str;
          continue;
        }

        const data = lastStr + str;
        lastStr = '';
        const normalizedData = data.replace(CRLF_RE, '\n');
        const dataBuffer = Buffer.from(normalizedData);
        size += dataBuffer.length;
        hash.update(dataBuffer);
      }
    }

    if (lastStr) {
      // NOTE: This does not normalize CRLFs or account for size — this matches
      // the original behavior. Fixing it would change the hash output.
      hash.update(Buffer.from(lastStr));
    }
  } finally {
    fs.closeSync(fd);
  }
  return `${size}:${hash.digest('hex')}`;
}

function _hashField(hash: crypto.Hash, header: string, value: string | Buffer | DataView) {
  hash.update(CTRL_SOH).update(header).update(CTRL_SOT).update(value).update(CTRL_ETX);
}
