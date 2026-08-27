import * as fs from 'fs';
import * as path from 'path';
import { SymlinkFollowMode } from './options';
import { UnscopedValidationError } from '../errors';
import { lit } from '../private/literal-string';

/**
 * Determines whether a symlink should be followed or not, based on a FollowMode.
 *
 * @param mode       the FollowMode.
 * @param sourceRoot the root of the source tree.
 * @param realPath   the real path of the target of the symlink.
 *
 * @returns true if the link should be followed.
 */
export function shouldFollow(mode: SymlinkFollowMode, sourceRoot: string, realPath: string): boolean {
  switch (mode) {
    case SymlinkFollowMode.ALWAYS:
      return fs.existsSync(realPath);
    case SymlinkFollowMode.EXTERNAL:
      return !_isInternal() && fs.existsSync(realPath);
    case SymlinkFollowMode.BLOCK_EXTERNAL:
      if (_isInternal()) {
        return fs.existsSync(realPath);
      } else {
        throw new UnscopedValidationError(lit`BundlingFileSymlinkForbidden`,
          `The file ${realPath} is an external symbolic link which is forbidden due to follow mode ${mode}. Set \`follow\` to a mode that will follow symlinks (ALWAYS or EXTERNAL) or emit a regular file`,
        );
      }
    case SymlinkFollowMode.NEVER:
      return false;
    default:
      throw new UnscopedValidationError(lit`UnsupportedFollowMode`, `Unsupported FollowMode: ${mode}`);
  }

  function _isInternal(): boolean {
    return isInternalPath(path.resolve(sourceRoot), path.resolve(realPath));
  }
}

export function isInternalPath(rootPath: string, targetPath: string): boolean {
  return rootPath === targetPath || targetPath.startsWith(rootPath + path.sep);
}

export function resolveLinkTarget(realPath: string, linkTarget: string): string {
  return path.isAbsolute(linkTarget)
    ? path.resolve(linkTarget)
    : path.resolve(path.dirname(realPath), linkTarget);
}
