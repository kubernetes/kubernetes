define(['./isKind'], function (isKind) {
    /**
     */
    function isRegExp(val) {
        return isKind(val, 'RegExp');
    }
    return isRegExp;
});
