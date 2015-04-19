define(function () {

    /**
     * Set query string parameter value
     */
    function setParam(url, paramName, value){
        url = url || '';

        var re = new RegExp('(\\?|&)'+ paramName +'=[^&]*' );
        var param = paramName +'='+ encodeURIComponent( value );

        if ( re.test(url) ) {
            return url.replace(re, '$1'+ param);
        } else {
            if (url.indexOf('?') === -1) {
                url += '?';
            }
            if (url.indexOf('=') !== -1) {
                url += '&';
            }
            return url + param;
        }

    }

    return setParam;

});
