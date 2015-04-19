(function() {

    console = function() {
        var stdout = java.lang.System.out;
        var stderr = java.lang.System.err;

        function doLog(out, type) {
            return function() {
                var args = java.lang.reflect.Array.newInstance(java.lang.Object, arguments.length - 1);
                var format = arguments[0];
                var conversionIndex = 0;
                // need to look for %d (integer) conversions because in Javascript all numbers are doubles
                for (var i = 1; i < arguments.length; i++) {
                    var arg = arguments[i];
                    if (conversionIndex != -1) {
                        conversionIndex = format.indexOf('%', conversionIndex);
                    }
                    if (conversionIndex >= 0 && conversionIndex < format.length) {
                        var conversion = format.charAt(conversionIndex + 1);
                        if (conversion === 'd' && typeof arg === 'number') {
                            arg = new java.lang.Integer(new java.lang.Double(arg).intValue());
                        }
                        conversionIndex++;
                    }
                    args[i-1] = arg;
                }
                try {
                    out.println(type + java.lang.String.format(format, args));
                } catch(ex) {
                    stderr.println(ex);
                }
            }
        }
        return {
            log: doLog(stdout, ''),
            info: doLog(stdout, 'INFO: '),
            error: doLog(stderr, 'ERROR: '),
            warn: doLog(stderr, 'WARN: ')
        };
    }();

    less.modules = {};

    less.modules.path = {
        join: function() {
            var parts = [];
            for (i in arguments) {
                parts = parts.concat(arguments[i].split(/\/|\\/));
            }
            var result = [];
            for (i in parts) {
                var part = parts[i];
                if (part === '..' && result.length > 0 && result[result.length-1] !== '..') {
                    result.pop();
                } else if (part === '' && result.length > 0) {
                    // skip
                } else if (part !== '.') {
		    if (part.slice(-1)==='\\' || part.slice(-1)==='/') {
		      part = part.slice(0, -1);
		    }
                    result.push(part);
                }
            }
            return result.join('/');
        },
        dirname: function(p) {
            var path = p.split('/');
            path.pop();
            return path.join('/');
        },
        basename: function(p, ext) {
            var base = p.split('/').pop();
            if (ext) {
                var index = base.lastIndexOf(ext);
                if (base.length === index + ext.length) {
                    base = base.substr(0, index);
                }
            }
            return base;
        },
        extname: function(p) {
            var index = p.lastIndexOf('.');
            return index > 0 ? p.substring(index) : '';
        }
    };

    less.modules.fs = {
        readFileSync: function(name) {
            // read a file into a byte array
            var file = new java.io.File(name);
            var stream = new java.io.FileInputStream(file);
            var buffer = [];
            var c;
            while ((c = stream.read()) != -1) {
                buffer.push(c);
            }
            stream.close();
            return {
                length: buffer.length,
                toString: function(enc) {
                    if (enc === 'base64') {
                        return encodeBase64Bytes(buffer);
                    } else if (enc) {
                        return java.lang.String["(byte[],java.lang.String)"](buffer, enc);
                    } else {
                        return java.lang.String["(byte[])"](buffer);
                    }
                }
            };
        }
    };

    less.encoder = {
        encodeBase64: function(str) {
            return encodeBase64String(str);
        }
    };

    // ---------------------------------------------------------------------------------------------
    // private helper functions
    // ---------------------------------------------------------------------------------------------

    function encodeBase64Bytes(bytes) {
        // requires at least a JRE Platform 6 (or JAXB 1.0 on the classpath)
        return javax.xml.bind.DatatypeConverter.printBase64Binary(bytes)
    }
    function encodeBase64String(str) {
        return encodeBase64Bytes(new java.lang.String(str).getBytes());
    }

})();
