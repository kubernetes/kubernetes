'use strict';

// The zip file spec is at http://www.pkware.com/documents/casestudies/APPNOTE.TXT
// TODO: There is fair chunk of the spec that I have ignored. Need to add
// assertions everywhere to make sure that we are not dealing with a ZIP type
// that I haven't designed for. Things like spanning archives, non-DEFLATE
// compression, encryption, etc.
var fs = require('graceful-fs');
var Q = require('q');
var path = require('path');
var util = require('util');
var events = require('events');
var structures = require('./structures');
var signatures = require('./signatures');
var extractors = require('./extractors');
var FileDetails = require('./file-details');

var fstat = Q.denodeify(fs.fstat);
var read = Q.denodeify(fs.read);
var fopen = Q.denodeify(fs.open);

function DecompressZip(filename) {
    events.EventEmitter.call(this);

    this.filename = filename;
    this.stats = null;
    this.fd = null;
    this.chunkSize = 1024 * 1024; // Buffer up to 1Mb at a time
    this.dirCache = {};

    // When we need a resource, we should check if there is a promise for it
    // already and use that. If the promise is already fulfilled we don't do the
    // async work again and we get to queue up dependant tasks.
    this._p = {}; // _p instead of _promises because it is a lot easier to read
}

util.inherits(DecompressZip, events.EventEmitter);

DecompressZip.prototype.openFile = function () {
    return fopen(this.filename, 'r');
};

DecompressZip.prototype.closeFile = function () {
    if (this.fd) {
        fs.closeSync(this.fd);
        this.fd = null;
    }
};

DecompressZip.prototype.statFile = function (fd) {
    this.fd = fd;
    return fstat(fd);
};

DecompressZip.prototype.list = function () {
    var self = this;

    this.getFiles()
    .then(function (files) {
        var result = [];

        files.forEach(function (file) {
            result.push(file.path);
        });

        self.emit('list', result);
    })
    .fail(function (error) {
        self.emit('error', error);
    })
    .fin(self.closeFile.bind(self));

    return this;
};

DecompressZip.prototype.extract = function (options) {
    var self = this;

    options = options || {};
    options.path = options.path || '.';
    options.filter = options.filter || null;
    options.follow = !!options.follow;
    options.strip = +options.strip || 0;

    this.getFiles()
    .then(function (files) {
        var copies = [];

        if (options.filter) {
            files = files.filter(options.filter);
        }

        if (options.follow) {
            copies = files.filter(function (file) {
                return file.type === 'SymbolicLink';
            });
            files = files.filter(function (file) {
                return file.type !== 'SymbolicLink';
            });
        }

        if (options.strip) {
            files = files.map(function (file) {
                if (file.type !== 'Directory') {
                    // we don't use `path.sep` as we're using `/` in Windows too
                    var dir = file.parent.split('/');
                    var filename = file.filename;

                    if (options.strip > dir.length) {
                        throw new Error('You cannot strip more levels than there are directories');
                    } else {
                        dir = dir.slice(options.strip);
                    }

                    file.path = path.join(dir.join(path.sep), filename);
                    return file;
                }
            });
        }

        return self.extractFiles(files, options)
        .then(self.extractFiles.bind(self, copies, options));
    })
    .then(function (results) {
        self.emit('extract', results);
    })
    .fail(function (error) {
        self.emit('error', error);
    })
    .fin(self.closeFile.bind(self));

    return this;
};

// Utility methods
DecompressZip.prototype.getSearchBuffer = function (stats) {
    var size = Math.min(stats.size, this.chunkSize);
    this.stats = stats;
    return this.getBuffer(stats.size - size, stats.size);
};

DecompressZip.prototype.getBuffer = function (start, end) {
    var size = end - start;
    return read(this.fd, new Buffer(size), 0, size, start)
    .then(function (result) {
        return result[1];
    });
};

DecompressZip.prototype.findEndOfDirectory = function (buffer) {
    var index = buffer.length - 3;
    var chunk = '';

    // Apparently the ZIP spec is not very good and it is impossible to
    // guarantee that you have read a zip file correctly, or to determine
    // the location of the CD without hunting.
    // Search backwards through the buffer, as it is very likely to be near the
    // end of the file.
    while (index > Math.max(buffer.length - this.chunkSize, 0) && chunk !== signatures.END_OF_CENTRAL_DIRECTORY) {
        index--;
        chunk = buffer.readUInt32LE(index);
    }

    if (chunk !== signatures.END_OF_CENTRAL_DIRECTORY) {
        throw new Error('Could not find the End of Central Directory Record');
    }

    return buffer.slice(index);
};

// Directory here means the ZIP Central Directory, not a folder
DecompressZip.prototype.readDirectory = function (recordBuffer) {
    var record = structures.readEndRecord(recordBuffer);

    return this.getBuffer(record.directoryOffset, record.directoryOffset + record.directorySize)
    .then(structures.readDirectory.bind(null));
};

DecompressZip.prototype.getFiles = function () {
    if (!this._p.getFiles) {
        this._p.getFiles = this.openFile()
        .then(this.statFile.bind(this))
        .then(this.getSearchBuffer.bind(this))
        .then(this.findEndOfDirectory.bind(this))
        .then(this.readDirectory.bind(this))
        .then(this.readFileEntries.bind(this));
    }

    return this._p.getFiles;
};

DecompressZip.prototype.readFileEntries = function (directory) {
    var promises = [];
    var files = [];
    var self = this;

    directory.forEach(function (directoryEntry, index) {
        var start = directoryEntry.relativeOffsetOfLocalHeader;
        var end = Math.min(self.stats.size, start + structures.maxFileEntrySize);
        var fileDetails = new FileDetails(directoryEntry);

        var promise = self.getBuffer(start, end)
        .then(structures.readFileEntry.bind(null))
        .then(function (fileEntry) {
            var maxSize;

            if (fileDetails.compressedSize > 0) {
                maxSize = fileDetails.compressedSize;
            } else {
                maxSize = self.stats.size;

                if (index < directory.length - 1) {
                    maxSize = directory[index + 1].relativeOffsetOfLocalHeader;
                }

                maxSize -= start + fileEntry.entryLength;
            }

            fileDetails._offset = start + fileEntry.entryLength;
            fileDetails._maxSize = maxSize;

            self.emit('file', fileDetails);
            files[index] = fileDetails;
        });

        promises.push(promise);
    });

    return Q.all(promises)
    .then(function () {
        return files;
    });
};

DecompressZip.prototype.extractFiles = function (files, options, results) {
    var promises = [];
    var self = this;

    results = results || [];
    var fileIndex = 0;
    files.forEach(function (file) {
        var promise = self.extractFile(file, options)
        .then(function (result) {
            self.emit('progress', fileIndex++, files.length);
            results.push(result);
        });

        promises.push(promise);
    });

    return Q.all(promises)
    .then(function () {
        return results;
    });
};

DecompressZip.prototype.extractFile = function (file, options) {
    var destination = path.join(options.path, file.path);

    // Possible compression methods:
    //    0 - The file is stored (no compression)
    //    1 - The file is Shrunk
    //    2 - The file is Reduced with compression factor 1
    //    3 - The file is Reduced with compression factor 2
    //    4 - The file is Reduced with compression factor 3
    //    5 - The file is Reduced with compression factor 4
    //    6 - The file is Imploded
    //    7 - Reserved for Tokenizing compression algorithm
    //    8 - The file is Deflated
    //    9 - Enhanced Deflating using Deflate64(tm)
    //   10 - PKWARE Data Compression Library Imploding (old IBM TERSE)
    //   11 - Reserved by PKWARE
    //   12 - File is compressed using BZIP2 algorithm
    //   13 - Reserved by PKWARE
    //   14 - LZMA (EFS)
    //   15 - Reserved by PKWARE
    //   16 - Reserved by PKWARE
    //   17 - Reserved by PKWARE
    //   18 - File is compressed using IBM TERSE (new)
    //   19 - IBM LZ77 z Architecture (PFS)
    //   97 - WavPack compressed data
    //   98 - PPMd version I, Rev 1

    if (file.type === 'Directory') {
        return extractors.folder(file, destination, this);
    }

    if (file.type === 'File') {
        switch (file.compressionMethod) {
        case 0:
            return extractors.store(file, destination, this);

        case 8:
            return extractors.deflate(file, destination, this);

        default:
            throw new Error('Unsupported compression type');
        }
    }

    if (file.type === 'SymbolicLink') {
        if (options.follow) {
            return extractors.copy(file, destination, this, options.path);
        } else {
            return extractors.symlink(file, destination, this, options.path);
        }
    }

    throw new Error('Unsupported file type "' + file.type + '"');
};

module.exports = DecompressZip;
