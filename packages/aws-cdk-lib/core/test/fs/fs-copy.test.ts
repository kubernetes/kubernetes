import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { FileSystem, IgnoreMode, SymlinkFollowMode } from '../../lib/fs';

describe('fs copy', () => {
  let outdir: string;
  beforeEach(() => {
    outdir = fs.mkdtempSync(path.join(os.tmpdir(), 'copy-tests'));
  });

  afterEach(() => {
    fs.rmSync(outdir, { force: true, recursive: true });
  });

  test('Default: copies all files and subdirectories, with default follow mode is "External"', () => {
    // WHEN
    FileSystem.copyDirectory(path.join(__dirname, 'fixtures', 'test1'), outdir);

    // THEN
    expect(tree(outdir)).toEqual([
      'external-link.txt',
      'file1.txt',
      'local-link.txt => file1.txt',
      'subdir (D)',
      '    file2.txt',
      'subdir2 (D)',
      '    empty-subdir (D)',
      '        .hidden',
      '    subdir3 (D)',
      '        file3.txt',
      'subdir4 (D)',
      '    file4.txt',
      '    local-link4.txt => file4.txt',
    ]);
  });

  test('Always: follow all symlinks', () => {
    // WHEN
    FileSystem.copyDirectory(path.join(__dirname, 'fixtures', 'symlinks'), outdir, {
      follow: SymlinkFollowMode.ALWAYS,
    });

    // THEN
    expect(tree(outdir)).toEqual([
      'external-dir-link (D)',
      '    file2.txt',
      'external-link.txt',
      'indirect-external-link.txt',
      'local-dir-link (D)',
      '    file-in-subdir.txt',
      'local-link.txt',
      'normal-dir (D)',
      '    file-in-subdir.txt',
      'normal-file.txt',
    ]);
  });

  test('Never: do not follow all symlinks', () => {
    // WHEN
    FileSystem.copyDirectory(path.join(__dirname, 'fixtures', 'symlinks'), outdir, {
      follow: SymlinkFollowMode.NEVER,
    });

    // THEN
    expect(tree(outdir)).toEqual([
      'external-dir-link => ../test1/subdir',
      'external-link.txt => ../test1/subdir2/subdir3/file3.txt',
      'indirect-external-link.txt => external-link.txt',
      'local-dir-link => normal-dir',
      'local-link.txt => normal-file.txt',
      'normal-dir (D)',
      '    file-in-subdir.txt',
      'normal-file.txt',
    ]);
  });

  test('External: follow only external symlinks', () => {
    // WHEN
    FileSystem.copyDirectory(path.join(__dirname, 'fixtures', 'symlinks'), outdir, {
      follow: SymlinkFollowMode.EXTERNAL,
    });

    // THEN
    expect(tree(outdir)).toEqual([
      'external-dir-link (D)',
      '    file2.txt',
      'external-link.txt',
      'indirect-external-link.txt => external-link.txt',
      'local-dir-link => normal-dir',
      'local-link.txt => normal-file.txt',
      'normal-dir (D)',
      '    file-in-subdir.txt',
      'normal-file.txt',
    ]);
  });

  test('exclude', () => {
    // WHEN
    FileSystem.copyDirectory(path.join(__dirname, 'fixtures', 'test1'), outdir, {
      exclude: [
        '*',
        '!subdir2',
        '!subdir2/**/*',
        '.*',
      ],
    });

    // THEN
    expect(tree(outdir)).toEqual([
      'subdir2 (D)',
      '    empty-subdir (D)',
      '    subdir3 (D)',
      '        file3.txt',
    ]);
  });

  test('nested exclude with docker ignore mode', () => {
    // WHEN
    FileSystem.copyDirectory(path.join(__dirname, 'fixtures', 'test1'), outdir, {
      exclude: [
        '**',
        '!subdir2/subdir3/*.txt',
        '!subdir/file2.txt',
        'subdir',
        '!local-link.txt',
      ],
      ignoreMode: IgnoreMode.DOCKER,
    });

    // THEN
    expect(tree(outdir)).toEqual([
      'local-link.txt => file1.txt',
      'subdir2 (D)',
      '    subdir3 (D)',
      '        file3.txt',
    ]);
  });

  test('negated pattern inside subdirectory with git ignore mode', () => {
    // WHEN
    FileSystem.copyDirectory(path.join(__dirname, 'fixtures', 'test1'), outdir, {
      exclude: [
        '*',
        '!.hidden',
        '!*/',
      ],
      ignoreMode: IgnoreMode.GIT,
    });

    // THEN
    expect(tree(outdir)).toEqual([
      'subdir2 (D)',
      '    empty-subdir (D)',
      '        .hidden',
    ]);
  });
});

function tree(dir: string, depth = ''): string[] {
  const lines = new Array<string>();
  for (const file of fs.readdirSync(dir).sort()) {
    const filePath = path.join(dir, file);
    const stat = fs.lstatSync(filePath);
    if (stat.isSymbolicLink()) {
      const linkDest = fs.readlinkSync(filePath);
      lines.push(depth + file + ' => ' + linkDest);
    } else if (stat.isDirectory()) {
      lines.push(depth + file + ' (D)');
      lines.push(...tree(filePath, depth + '    '));
    } else {
      lines.push(depth + file);
    }
  }
  return lines;
}
