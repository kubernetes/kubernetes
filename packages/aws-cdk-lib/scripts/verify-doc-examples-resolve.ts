/**
 * Verify that every `aws-cdk-lib/<subpath>` specifier used in a documentation example is a
 * subpath that package.json actually exports.
 *
 * Examples are meant to be copy-pasted, so a specifier naming a module we do not export leaves
 * the reader with an import that cannot resolve. Rosetta compiles `ts` fences and would catch a
 * bad `import` statement, but a specifier can also appear as a plain string -- a
 * `setupFilesAfterEnv` entry, for instance -- and a string is not an import, so nothing checks it.
 *
 * What gets checked: fenced code blocks in `.md` files, and fenced code blocks inside doc
 * comments in `.ts` sources. Anything outside a fence is left alone, because `@see` tags and
 * prose legitimately mention jsii type names and repository file paths, neither of which is a
 * module specifier.
 */
import * as fs from 'fs';
import * as path from 'path';

const SPECIFIER = /aws-cdk-lib\/[A-Za-z0-9._/-]+/g;
const SKIP_DIRS = new Set(['node_modules', 'test', 'build-tools']);

interface Mention {
  readonly specifier: string;
  readonly file: string;
  readonly line: number;
}

const pkgDir = path.resolve(__dirname, '..');
const pkg = JSON.parse(fs.readFileSync(path.join(pkgDir, 'package.json'), 'utf-8'));
const exportedSubpaths = new Set(Object.keys(pkg.exports ?? {}));

function documentationFiles(dir: string, into: string[] = []): string[] {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (!SKIP_DIRS.has(entry.name)) {
        documentationFiles(full, into);
      }
    } else if (entry.isFile() && isDocumentationFile(entry.name)) {
      into.push(full);
    }
  }
  return into;
}

function isDocumentationFile(name: string): boolean {
  if (name.endsWith('.md')) {
    return true;
  }
  return name.endsWith('.ts') && !name.endsWith('.generated.ts') && !name.endsWith('.d.ts');
}

/**
 * Collect the specifiers mentioned inside fenced code blocks.
 *
 * For markdown the whole file counts; for TypeScript only fences inside a doc comment do.
 */
function mentionsInExamples(file: string): Mention[] {
  const markdown = file.endsWith('.md');
  const mentions: Mention[] = [];

  let inDocComment = markdown;
  let inFence = false;

  fs.readFileSync(file, 'utf-8').split('\n').forEach((text, index) => {
    if (!markdown && text.includes('/**')) {
      inDocComment = true;
      inFence = false;
    }

    // A fence delimiter opens or closes a block and never carries a specifier itself. In a doc
    // comment the delimiter is preceded by the comment's leading asterisk.
    const isFenceDelimiter = markdown ? /^\s*```/.test(text) : /^\s*\*?\s*```/.test(text);

    if (inDocComment && isFenceDelimiter) {
      inFence = !inFence;
    } else if (inDocComment && inFence) {
      for (const match of text.matchAll(SPECIFIER)) {
        mentions.push({
          specifier: match[0].replace(/\.+$/, ''),
          file: path.relative(pkgDir, file),
          line: index + 1,
        });
      }
    }

    if (!markdown && text.includes('*/')) {
      inDocComment = false;
      inFence = false;
    }
  });

  return mentions;
}

const mentions = documentationFiles(pkgDir).flatMap(mentionsInExamples);

// A scan that matches nothing would report success no matter how broken the docs are, so treat
// finding no specifiers at all as the scan itself being broken.
if (mentions.length === 0) {
  console.error('Found no `aws-cdk-lib/...` specifiers in any documentation example.');
  console.error('That is not plausible -- this check is no longer looking in the right places.');
  process.exitCode = 1;
} else {
  const dangling = mentions.filter((m) => !exportedSubpaths.has(`.${m.specifier.slice('aws-cdk-lib'.length)}`));

  if (dangling.length > 0) {
    console.error(`Found ${dangling.length} documentation example(s) naming a module that is not exported:\n`);
    for (const m of dangling) {
      console.error(`  ${m.file}:${m.line} -> ${m.specifier}`);
    }
    console.error('\nEither correct the specifier, or add the subpath to the "exports" map in package.json.');
    process.exitCode = 1;
  } else {
    console.log(`All ${mentions.length} documentation example specifier(s) resolve to an exported subpath.`);
  }
}
