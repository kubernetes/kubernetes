function createError(msg, code) {
    var err = new Error(msg);
    err.code = code;

    return err;
}

module.exports = createError;
