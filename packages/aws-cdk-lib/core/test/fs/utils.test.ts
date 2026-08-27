import fs from 'fs';
import path from 'path';
import { ImportMock } from 'ts-mock-imports';
import { SymlinkFollowMode } from '../../lib/fs';
import * as util from '../../lib/fs/utils';

describe('utils', () => {
  describe('shouldFollow', () => {
    describe('always', () => {
      test('follows internal', () => {
        const sourceRoot = path.join('source', 'root');
        const linkTarget = path.join(sourceRoot, 'referent');

        const mockFsExists = ImportMock.mockFunction(fs, 'existsSync', true);
        try {
          expect(util.shouldFollow(SymlinkFollowMode.ALWAYS, sourceRoot, linkTarget)).toEqual(true);
          expect(mockFsExists.calledOnceWith(linkTarget)).toEqual(true);
        } finally {
          mockFsExists.restore();
        }
      });

      test('follows external', () => {
        const sourceRoot = path.join('source', 'root');
        const linkTarget = path.join('alternate', 'referent');
        const mockFsExists = ImportMock.mockFunction(fs, 'existsSync', true);
        try {
          expect(util.shouldFollow(SymlinkFollowMode.ALWAYS, sourceRoot, linkTarget)).toEqual(true);
          expect(mockFsExists.calledOnceWith(linkTarget)).toEqual(true);
        } finally {
          mockFsExists.restore();
        }
      });

      test('does not follow internal when the referent does not exist', () => {
        const sourceRoot = path.join('source', 'root');
        const linkTarget = path.join(sourceRoot, 'referent');
        const mockFsExists = ImportMock.mockFunction(fs, 'existsSync', false);
        try {
          expect(util.shouldFollow(SymlinkFollowMode.ALWAYS, sourceRoot, linkTarget)).toEqual(false);
          expect(mockFsExists.calledOnceWith(linkTarget)).toEqual(true);
        } finally {
          mockFsExists.restore();
        }
      });

      test('does not follow external when the referent does not exist', () => {
        const sourceRoot = path.join('source', 'root');
        const linkTarget = path.join('alternate', 'referent');
        const mockFsExists = ImportMock.mockFunction(fs, 'existsSync', false);
        try {
          expect(util.shouldFollow(SymlinkFollowMode.ALWAYS, sourceRoot, linkTarget)).toEqual(false);
          expect(mockFsExists.calledOnceWith(linkTarget)).toEqual(true);
        } finally {
          mockFsExists.restore();
        }
      });
    });

    describe('external', () => {
      test('does not follow internal', () => {
        const sourceRoot = path.join('source', 'root');
        const linkTarget = path.join(sourceRoot, 'referent');
        const mockFsExists = ImportMock.mockFunction(fs, 'existsSync');
        try {
          expect(util.shouldFollow(SymlinkFollowMode.EXTERNAL, sourceRoot, linkTarget)).toEqual(false);
          expect(mockFsExists.notCalled).toEqual(true);
        } finally {
          mockFsExists.restore();
        }
      });

      test('follows external', () => {
        const sourceRoot = path.join('source', 'root');
        const linkTarget = path.join('alternate', 'referent');
        const mockFsExists = ImportMock.mockFunction(fs, 'existsSync', true);
        try {
          expect(util.shouldFollow(SymlinkFollowMode.EXTERNAL, sourceRoot, linkTarget)).toEqual(true);
          expect(mockFsExists.calledOnceWith(linkTarget)).toEqual(true);
        } finally {
          mockFsExists.restore();
        }
      });

      test('follows a sibling that shares a name prefix with the root', () => {
        // 'source/root-sibling' is NOT inside 'source/root', even though its path
        // string starts with the root string. It must be treated as external.
        const sourceRoot = path.join('source', 'root');
        const linkTarget = path.join('source', 'root-sibling', 'referent');
        const mockFsExists = ImportMock.mockFunction(fs, 'existsSync', true);
        try {
          expect(util.shouldFollow(SymlinkFollowMode.EXTERNAL, sourceRoot, linkTarget)).toEqual(true);
          expect(mockFsExists.calledOnceWith(linkTarget)).toEqual(true);
        } finally {
          mockFsExists.restore();
        }
      });

      test('does not follow external when referent does not exist', () => {
        const sourceRoot = path.join('source', 'root');
        const linkTarget = path.join('alternate', 'referent');
        const mockFsExists = ImportMock.mockFunction(fs, 'existsSync', false);
        try {
          expect(util.shouldFollow(SymlinkFollowMode.EXTERNAL, sourceRoot, linkTarget)).toEqual(false);
          expect(mockFsExists.calledOnceWith(linkTarget)).toEqual(true);
        } finally {
          mockFsExists.restore();
        }
      });
    });

    describe('blockExternal', () => {
      test('follows internal', () => {
        const sourceRoot = path.join('source', 'root');
        const linkTarget = path.join(sourceRoot, 'referent');
        const mockFsExists = ImportMock.mockFunction(fs, 'existsSync', true);
        try {
          expect(util.shouldFollow(SymlinkFollowMode.BLOCK_EXTERNAL, sourceRoot, linkTarget)).toEqual(true);
          expect(mockFsExists.calledOnceWith(linkTarget)).toEqual(true);
        } finally {
          mockFsExists.restore();
        }
      });

      test('does not follow internal when referent does not exist', () => {
        const sourceRoot = path.join('source', 'root');
        const linkTarget = path.join(sourceRoot, 'referent');
        const mockFsExists = ImportMock.mockFunction(fs, 'existsSync', false);
        try {
          expect(util.shouldFollow(SymlinkFollowMode.BLOCK_EXTERNAL, sourceRoot, linkTarget)).toEqual(false);
          expect(mockFsExists.calledOnceWith(linkTarget)).toEqual(true);
        } finally {
          mockFsExists.restore();
        }
      });

      test('throws on external', () => {
        const sourceRoot = path.join('source', 'root');
        const linkTarget = path.join('alternate', 'referent');
        const mockFsExists = ImportMock.mockFunction(fs, 'existsSync');
        try {
          expect(() => util.shouldFollow(SymlinkFollowMode.BLOCK_EXTERNAL, sourceRoot, linkTarget))
            .toThrow(/external symbolic link which is forbidden/);
        } finally {
          mockFsExists.restore();
        }
      });

      test('throws on a sibling that shares a name prefix with the root', () => {
        // 'source/root-sibling' is external to 'source/root' despite the shared
        // string prefix, so BLOCK_EXTERNAL must reject it.
        const sourceRoot = path.join('source', 'root');
        const linkTarget = path.join('source', 'root-sibling', 'referent');
        const mockFsExists = ImportMock.mockFunction(fs, 'existsSync');
        try {
          expect(() => util.shouldFollow(SymlinkFollowMode.BLOCK_EXTERNAL, sourceRoot, linkTarget))
            .toThrow(/external symbolic link which is forbidden/);
        } finally {
          mockFsExists.restore();
        }
      });
    });

    describe('never', () => {
      test('does not follow internal', () => {
        const sourceRoot = path.join('source', 'root');
        const linkTarget = path.join(sourceRoot, 'referent');
        const mockFsExists = ImportMock.mockFunction(fs, 'existsSync');
        try {
          expect(util.shouldFollow(SymlinkFollowMode.NEVER, sourceRoot, linkTarget)).toEqual(false);
          expect(mockFsExists.notCalled).toEqual(true);
        } finally {
          mockFsExists.restore();
        }
      });

      test('does not follow external', () => {
        const sourceRoot = path.join('source', 'root');
        const linkTarget = path.join('alternate', 'referent');
        const mockFsExists = ImportMock.mockFunction(fs, 'existsSync');
        try {
          expect(util.shouldFollow(SymlinkFollowMode.NEVER, sourceRoot, linkTarget)).toEqual(false);
          expect(mockFsExists.notCalled).toEqual(true);
        } finally {
          mockFsExists.restore();
        }
      });
    });
  });

  describe('isInternalPath', () => {
    const root = path.resolve(path.join('source', 'root'));

    test('a descendant of the root is internal', () => {
      expect(util.isInternalPath(root, path.join(root, 'child', 'file.txt'))).toEqual(true);
    });

    test('a sibling sharing a name prefix with the root is external', () => {
      // 'source/root-sibling' string-prefixes 'source/root' but is NOT inside it.
      expect(util.isInternalPath(root, root + '-sibling')).toEqual(false);
      expect(util.isInternalPath(root, root + '-sibling' + path.sep + 'file.txt')).toEqual(false);
    });

    test('an unrelated path is external', () => {
      expect(util.isInternalPath(root, path.resolve(path.join('source', 'elsewhere', 'file.txt')))).toEqual(false);
    });
  });

  describe('resolveLinkTarget', () => {
    test('an absolute link target is resolved as-is', () => {
      const realPath = path.join('source', 'root', 'link');
      const linkTarget = path.resolve(path.join('somewhere', 'else', 'referent'));

      expect(util.resolveLinkTarget(realPath, linkTarget)).toEqual(path.resolve(linkTarget));
    });

    test('a relative link target is resolved against the directory of the link', () => {
      const realPath = path.join('source', 'root', 'link');
      const linkTarget = 'referent';

      // Resolved relative to the link's directory ('source/root'), not the cwd.
      expect(util.resolveLinkTarget(realPath, linkTarget)).toEqual(
        path.resolve(path.join('source', 'root'), 'referent'),
      );
    });

    test('a relative link target with parent segments is normalized', () => {
      const realPath = path.join('source', 'root', 'nested', 'link');
      const linkTarget = path.join('..', 'sibling', 'referent');

      expect(util.resolveLinkTarget(realPath, linkTarget)).toEqual(
        path.resolve(path.join('source', 'root', 'sibling', 'referent')),
      );
    });
  });
});
