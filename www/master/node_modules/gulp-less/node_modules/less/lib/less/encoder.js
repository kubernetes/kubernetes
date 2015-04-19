// base64 encoder implementation for node
exports.encodeBase64 = function(str) {
    return new Buffer(str).toString('base64');
};
