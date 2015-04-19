'use strict';
var binary = require('binary');

var convertDateTime = function (dosDate, dosTime) {
    var year = ((dosDate >> 9) & 0x7F) + 1980;
    var month = (dosDate >> 5) & 0x0F;
    var day = dosDate & 0x1F;

    var hour = (dosTime >> 11);
    var minute = (dosTime >> 5) & 0x3F;
    var second = (dosTime & 0x1F) * 2;

    var result = new Date(year, month - 1, day, hour, minute, second, 0);

    return result;
};

var convertGeneralPurposeFlags = function (value) {
    var bits = [];

    for (var i = 0; i < 16; i++) {
        bits[i] = (value >> i) & 1;
    }

    return {
        encrypted: !!bits[0],
        compressionFlag1: !!bits[1],
        compressionFlag2: !!bits[2],
        useDataDescriptor: !!bits[3],
        enhancedDeflating: !!bits[4],
        compressedPatched: !!bits[5],
        strongEncryption: !!bits[6],
        utf8: !!bits[11],
        encryptedCD: !!bits[13]
    };
};

var parseExternalFileAttributes = function (externalAttributes, platform) {
    var types = {
        // In theory, any of these could be set. Realistically, though, it will
        // be regular, directory or symlink
        1: 'NamedPipe',
        2: 'Character',
        4: 'Directory',
        6: 'Block',
        8: 'File',
        10: 'SymbolicLink',
        12: 'Socket'
    };

    switch (platform) {

    case 3: // Unix
        return {
            platform: 'Unix',
            type: types[(externalAttributes >> 60) & 0x0F],
            mode: (externalAttributes >> 48) & 0xFFF
        };

    // case 0: // MSDOS
    default:
        if (platform !== 0) {
            console.warn('Possibly unsupported ZIP platform type, ' + platform);
        }

        var attribs = {
            A: (externalAttributes >> 5) & 0x01,
            D: (externalAttributes >> 4) & 0x01,
            V: (externalAttributes >> 3) & 0x01,
            S: (externalAttributes >> 2) & 0x01,
            H: (externalAttributes >> 1) & 0x01,
            R: externalAttributes & 0x01
        };

        // With no better guidance we'll make the default permissions ugo+r
        var mode = parseInt('0444', 8);

        if (attribs.D) {
            mode |= parseInt('0111', 8); // Set the execute bit
        }

        if (!attribs.R) {
            mode |= parseInt('0222', 8); // Set the write bit
        }

        mode &= ~process.umask();

        return {
            platform: 'DOS',
            type: attribs.D ? 'Directory' : 'File',
            mode: mode
        };
    }
};

var readEndRecord = function (buffer) {
    var data = binary.parse(buffer)
    .word32lu('signature')
    .word16lu('diskNumber')
    .word16lu('directoryStartDisk')
    .word16lu('directoryEntryCountDisk')
    .word16lu('directoryEntryCount')
    .word32lu('directorySize')
    .word32lu('directoryOffset')
    .word16lu('commentLength')
    .buffer('comment', 'commentLength')
    .vars;

    data.comment = data.comment.toString();

    return data;
};

var directorySort = function (a, b) {
    return a.relativeOffsetOfLocalHeader - b.relativeOffsetOfLocalHeader;
};

var readDirectory = function (buffer) {
    var directory = [];
    var current;
    var index = 0;

    while (index < buffer.length) {
        current = binary.parse(buffer.slice(index, index + 46))
        .word32lu('signature')
        .word8lu('creatorSpecVersion')
        .word8lu('creatorPlatform')
        .word8lu('requiredSpecVersion')
        .word8lu('requiredPlatform')
        .word16lu('generalPurposeBitFlag')
        .word16lu('compressionMethod')
        .word16lu('lastModFileTime')
        .word16lu('lastModFileDate')
        .word32lu('crc32')
        .word32lu('compressedSize')
        .word32lu('uncompressedSize')
        .word16lu('fileNameLength')
        .word16lu('extraFieldLength')
        .word16lu('fileCommentLength')
        .word16lu('diskNumberStart')
        .word16lu('internalFileAttributes')
        .word32lu('externalFileAttributes')
        .word32lu('relativeOffsetOfLocalHeader')
        .vars;

        index += 46;

        current.generalPurposeFlags = convertGeneralPurposeFlags(current.generalPurposeBitFlag);
        current.fileAttributes = parseExternalFileAttributes(current.externalFileAttributes, current.creatorPlatform);

        current.modifiedTime = convertDateTime(current.lastModFileDate, current.lastModFileTime);
        current.fileName = current.extraField = current.fileComment = '';
        current.headerLength = 46 + current.fileNameLength + current.extraFieldLength + current.fileCommentLength;

        if (current.fileNameLength > 0) {
            current.fileName = buffer.slice(index, index + current.fileNameLength).toString();
            index += current.fileNameLength;
        }

        if (current.extraFieldLength > 0) {
            current.extraField = buffer.slice(index, index + current.extraFieldLength).toString();
            index += current.extraFieldLength;
        }

        if (current.fileCommentLength > 0) {
            current.fileComment = buffer.slice(index, index + current.fileCommentLength).toString();
            index += current.fileCommentLength;
        }

        if (current.fileAttributes.type !== 'Directory' && current.fileName.substr(-1) === '/') {
            // TODO: check that this is a reasonable check
            current.fileAttributes.type = 'Directory';
        }

        directory.push(current);
    }

    directory.sort(directorySort);

    return directory;
};

var readFileEntry = function (buffer) {
    var index = 0;

    var fileEntry = binary.parse(buffer.slice(index, 30))
    .word32lu('signature')
    .word16lu('versionNeededToExtract')
    .word16lu('generalPurposeBitFlag')
    .word16lu('compressionMethod')
    .word16lu('lastModFileTime')
    .word16lu('lastModFileDate')
    .word32lu('crc32')
    .word32lu('compressedSize')
    .word32lu('uncompressedSize')
    .word16lu('fileNameLength')
    .word16lu('extraFieldLength')
    .vars;

    index += 30;

    fileEntry.fileName = fileEntry.extraField = '';

    fileEntry.entryLength = 30 + fileEntry.fileNameLength + fileEntry.extraFieldLength;

    if (fileEntry.entryLength > structures.maxFileEntrySize) {
        throw new Error('File entry unexpectedly large: ' + fileEntry.entryLength + ' (max: ' + structures.maxFileEntrySize + ')');
    }

    if (fileEntry.fileNameLength > 0) {
        fileEntry.fileName = buffer.slice(index, index + fileEntry.fileNameLength).toString();
        index += fileEntry.fileNameLength;
    }

    if (fileEntry.extraFieldLength > 0) {
        fileEntry.extraField = buffer.slice(index, index + fileEntry.extraFieldLength).toString();
        index += fileEntry.extraFieldLength;
    }

    return fileEntry;
};

var structures = module.exports = {
    readEndRecord: readEndRecord,
    readDirectory: readDirectory,
    readFileEntry: readFileEntry,
    maxFileEntrySize: 4096
};
