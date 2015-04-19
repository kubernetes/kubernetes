var stream = require('stream');
if (!stream.Readable) {
    var stream = require('readable-stream');
}
var fs = require('graceful-fs');
var Q = require('q');
var path = require('path');
var zlib = require('zlib');
var touch = Q.denodeify(require('touch'));
var mkpath = Q.denodeify(require('mkpath'));
var writeFile = Q.denodeify(fs.writeFile);
var inflateRaw = Q.denodeify(zlib.inflateRaw);
var symlink = Q.denodeify(fs.symlink);
var stat = Q.denodeify(fs.stat);

// Use a cache of promises for building the directory tree. This allows us to
// correctly queue up file extractions for after their path has been created,
// avoid trying to create the path twice and still be async.
var mkdir = function (dir, cache) {
    dir = path.normalize(path.resolve(process.cwd(), dir) + path.sep);

    if (!cache[dir]) {
        var parent;

        if (fs.existsSync(dir)) {
            parent = new Q();
        } else {
            parent = mkdir(path.dirname(dir), cache);
        }

        cache[dir] = parent.then(function () {
            return mkpath(dir);
        });
    }

    return cache[dir];
};

// Utility methods for writing output files
var extractors = {
    folder: function (folder, destination, zip) {
        return mkdir(destination, zip.dirCache)
        .then(function () {
            return {folder: folder.path};
        });
    },
    store: function (file, destination, zip) {
        var writer;

        if (file.uncompressedSize === 0) {
            writer = touch.bind(null, destination);
        } else if (file.uncompressedSize <= zip.chunkSize) {
            writer = function () {
                return zip.getBuffer(file._offset, file._offset + file.uncompressedSize)
                .then(writeFile.bind(null, destination));
            };
        } else {
            var input = new stream.Readable();
            input.wrap(fs.createReadStream(zip.filename, {start: file._offset, end: file._offset + file.uncompressedSize - 1}));
            writer = pipePromise.bind(null, input, destination);
        }

        return mkdir(path.dirname(destination), zip.dirCache)
        .then(writer)
        .then(function () {
            return {stored: file.path};
        });
    },
    deflate: function (file, destination, zip) {
        // For Deflate you don't actually need to specify the end offset - and
        // in fact many ZIP files don't include compressed file sizes for
        // Deflated files so we don't even know what the end offset is.

        return mkdir(path.dirname(destination), zip.dirCache)
        .then(function () {
            if (file._maxSize <= zip.chunkSize) {
                return zip.getBuffer(file._offset, file._offset + file._maxSize)
                .then(inflateRaw)
                .then(function (buffer) {
                    return writeFile(destination, buffer);
                });
            } else {
                // For node 0.8 we need to create the Zlib stream and attach
                // handlers in the same tick of the event loop, which is why we do
                // the creation in here
                var input = new stream.Readable();
                input.wrap(fs.createReadStream(zip.filename, {start: file._offset}));
                var inflater = input.pipe(zlib.createInflateRaw({highWaterMark: 32 * 1024}));

                return pipePromise(inflater, destination);
            }
        })
        .then(function () {
            return {deflated: file.path};
        });
    },
    symlink: function (file, destination, zip, basePath) {
        var parent = path.dirname(destination);
        return mkdir(parent, zip.dirCache)
        .then(function () {
            return getLinkLocation(file, destination, zip, basePath);
        })
        .then(function (linkTo) {
            return symlink(path.resolve(parent, linkTo), destination)
            .then(function () {
                return {symlink: file.path, linkTo: linkTo};
            });
        });
    },
    // Make a shallow copy of the file/directory this symlink points to instead
    // of actually creating a link
    copy: function (file, destination, zip, basePath) {
        var type;
        var parent = path.dirname(destination);

        return mkdir(parent, zip.dirCache)
        .then(function () {
            return getLinkLocation(file, destination, zip, basePath);
        })
        .then(function (linkTo) {
            return stat(path.resolve(parent, linkTo))
            .then(function (stats) {
                if (stats.isFile()) {
                    type = 'File';
                    var input = new stream.Readable();
                    input.wrap(fs.createReadStream(path.resolve(parent, linkTo)));
                    return pipePromise(input, destination);
                } else if (stats.isDirectory()) {
                    type = 'Directory';
                    return mkdir(destination, zip.dirCache);
                } else {
                    throw new Error('Could not follow symlink to unknown file type');
                }
            })
            .then(function () {
                return {copy: file.path, original: linkTo, type: type};
            });
        });
    }
};

var getLinkLocation = function (file, destination, zip, basePath) {
    var parent = path.dirname(destination);
    return zip.getBuffer(file._offset, file._offset + file.uncompressedSize)
    .then(function (buffer) {
        var linkTo = buffer.toString();
        var fullLink = path.resolve(parent, linkTo);

        if (path.relative(basePath, fullLink).slice(0, 2) === '..') {
            throw new Error('Symlink links outside archive');
        }

        return linkTo;
    });
};

var pipePromise = function (input, destination) {
    var deferred = Q.defer();
    var output = fs.createWriteStream(destination);
    var errorHandler = function (error) {
        deferred.reject(error);
    };

    input.on('error', errorHandler);
    output.on('error', errorHandler);

    // For node 0.8 we can't just use the 'finish' event of the pipe
    input.on('end', function () {
        output.end(function () {
            deferred.resolve();
        });
    });

    input.pipe(output, {end: false});

    return deferred.promise;
};

module.exports = extractors;
