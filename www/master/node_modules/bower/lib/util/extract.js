var path = require('path');
var fs = require('graceful-fs');
var zlib = require('zlib');
var DecompressZip = require('decompress-zip');
var tar = require('tar-fs');
var Q = require('q');
var mout = require('mout');
var junk = require('junk');
var createError = require('./createError');

// This forces the default chunk size to something small in an attempt
// to avoid issue #314
zlib.Z_DEFAULT_CHUNK = 1024 * 8;

var extractors;
var extractorTypes;

extractors = {
    '.zip': extractZip,
    '.tar': extractTar,
    '.tar.gz': extractTarGz,
    '.tgz': extractTarGz,
    '.gz': extractGz,
    'application/zip': extractZip,
    'application/x-zip': extractZip,
    'application/x-zip-compressed': extractZip,
    'application/x-tar': extractTar,
    'application/x-tgz': extractTarGz,
    'application/x-gzip': extractGz
};

extractorTypes = Object.keys(extractors);

function extractZip(archive, dst) {
    var deferred = Q.defer();

    new DecompressZip(archive)
    .on('error', deferred.reject)
    .on('extract', deferred.resolve.bind(deferred, dst))
    .extract({
        path: dst,
        follow: false,           // Do not follow symlinks (#699)
        filter: filterSymlinks   // Filter symlink files
    });

    return deferred.promise;
}

function extractTar(archive, dst) {
    var deferred = Q.defer();

    fs.createReadStream(archive)
    .on('error', deferred.reject)
    .pipe(tar.extract(dst, {
        ignore: isSymlink, // Filter symlink files
        dmode: 0555,       // Ensure dirs are readable
        fmode: 0444        // Ensure files are readable
    }))
    .on('error', deferred.reject)
    .on('finish', deferred.resolve.bind(deferred, dst));

    return deferred.promise;
}

function extractTarGz(archive, dst) {
    var deferred = Q.defer();

    fs.createReadStream(archive)
    .on('error', deferred.reject)
    .pipe(zlib.createGunzip())
    .on('error', deferred.reject)
    .pipe(tar.extract(dst, {
        ignore: isSymlink, // Filter symlink files
        dmode: 0555,       // Ensure dirs are readable
        fmode: 0444        // Ensure files are readable
    }))
    .on('error', deferred.reject)
    .on('finish', deferred.resolve.bind(deferred, dst));

    return deferred.promise;
}

function extractGz(archive, dst) {
    var deferred = Q.defer();

    fs.createReadStream(archive)
    .on('error', deferred.reject)
    .pipe(zlib.createGunzip())
    .on('error', deferred.reject)
    .pipe(fs.createWriteStream(dst))
    .on('error', deferred.reject)
    .on('close', deferred.resolve.bind(deferred, dst));

    return deferred.promise;
}

function isSymlink(entry) {
    return entry.type === 'SymbolicLink';
}

function filterSymlinks(entry) {
    return entry.type !== 'SymbolicLink';
}

function getExtractor(archive) {
    // Make the archive lower case to match against the types
    // This ensures that upper-cased extensions work
    archive = archive.toLowerCase();

    var type = mout.array.find(extractorTypes, function (type) {
        return mout.string.endsWith(archive, type);
    });

    return type ? extractors[type] : null;
}

function isSingleDir(dir) {
    return Q.nfcall(fs.readdir, dir)
    .then(function (files) {
        var singleDir;

        // Remove any OS specific files from the files array
        // before checking its length
        files = files.filter(junk.isnt);

        if (files.length !== 1) {
            return false;
        }

        singleDir = path.join(dir, files[0]);

        return Q.nfcall(fs.stat, singleDir)
        .then(function (stat) {
            return stat.isDirectory() ? singleDir : false;
        });
    });
}

function moveSingleDirContents(dir) {
    var destDir = path.dirname(dir);

    return Q.nfcall(fs.readdir, dir)
    .then(function (files) {
        var promises;

        promises = files.map(function (file) {
            var src = path.join(dir, file);
            var dst = path.join(destDir, file);

            return Q.nfcall(fs.rename, src, dst);
        });

        return Q.all(promises);
    })
    .then(function () {
        return Q.nfcall(fs.rmdir, dir);
    });
}

// -----------------------------

function canExtract(src, mimeType) {
    if (mimeType && mimeType !== 'application/octet-stream') {
        return !!getExtractor(mimeType);
    }

    return !!getExtractor(src);
}

// Available options:
// - keepArchive: true to keep the archive afterwards (defaults to false)
// - keepStructure: true to keep the extracted structure unchanged (defaults to false)
function extract(src, dst, opts) {
    var extractor;
    var promise;

    opts = opts || {};
    extractor = getExtractor(src);

    // Try to get extractor from mime type
    if (!extractor && opts.mimeType) {
        extractor = getExtractor(opts.mimeType);
    }

    // If extractor is null, then the archive type is unknown
    if (!extractor) {
        return Q.reject(createError('File ' + src + ' is not a known archive', 'ENOTARCHIVE'));
    }

    // Check archive file size
    promise = Q.nfcall(fs.stat, src)
    .then(function (stat) {
        if (stat.size <= 8) {
            throw createError('File ' + src + ' is an invalid archive', 'ENOTARCHIVE');
        }

        // Extract archive
        return extractor(src, dst);
    });

    // TODO: There's an issue here if the src and dst are the same and
    //       The zip name is the same as some of the zip file contents
    //       Maybe create a temp directory inside dst, unzip it there,
    //       unlink zip and then move contents

    // Remove archive
    if (!opts.keepArchive) {
        promise = promise
        .then(function () {
            return Q.nfcall(fs.unlink, src);
        });
    }

    // Move contents if a single directory was extracted
    if (!opts.keepStructure) {
        promise = promise
        .then(function () {
            return isSingleDir(dst);
        })
        .then(function (singleDir) {
            return singleDir ? moveSingleDirContents(singleDir) : null;
        });
    }

    // Resolve promise to the dst dir
    return promise.then(function () {
        return dst;
    });
}

module.exports = extract;
module.exports.canExtract = canExtract;
