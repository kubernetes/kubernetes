define(['./isKind'], function (isKind) {
    /**
     */
    function isString(val) {
        return isKind(val, 'String');
    }
    return isString;
});
