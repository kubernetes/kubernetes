// Objects with this prototype are used as the public representation of a file
var path = require('path');

var FileDetails = function (directoryEntry) {
    // TODO: Add 'extra field' support

    this._offset = 0;
    this._maxSize = 0;

    this.parent = path.dirname(directoryEntry.fileName);
    this.filename = path.basename(directoryEntry.fileName);
    this.path = path.normalize(directoryEntry.fileName);

    this.type = directoryEntry.fileAttributes.type;
    this.mode = directoryEntry.fileAttributes.mode;
    this.compressionMethod = directoryEntry.compressionMethod;
    this.modified = directoryEntry.modifiedTime;
    this.crc32 = directoryEntry.crc32;
    this.compressedSize = directoryEntry.compressedSize;
    this.uncompressedSize = directoryEntry.uncompressedSize;
    this.comment = directoryEntry.fileComment;

    this.flags = {
        encrypted: directoryEntry.generalPurposeFlags.encrypted,
        compressionFlag1: directoryEntry.generalPurposeFlags.compressionFlag1,
        compressionFlag2: directoryEntry.generalPurposeFlags.compressionFlag2,
        useDataDescriptor: directoryEntry.generalPurposeFlags.useDataDescriptor,
        enhancedDeflating: directoryEntry.generalPurposeFlags.enhancedDeflating,
        compressedPatched: directoryEntry.generalPurposeFlags.compressedPatched,
        strongEncryption: directoryEntry.generalPurposeFlags.strongEncryption,
        utf8: directoryEntry.generalPurposeFlags.utf8,
        encryptedCD: directoryEntry.generalPurposeFlags.encryptedCD
    };

};

module.exports = FileDetails;
