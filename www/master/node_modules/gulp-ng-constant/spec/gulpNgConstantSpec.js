'use strict';

var ngConstant = require('../index');

describe('ngConstant.getConstants', function () {
    it('returns an array with the constants key and json string value', function () {
        var data = { constants: { hello: { foo: 'bar' } } };
        var result = ngConstant.getConstants(data);
        expect(result[0]).toEqual({name: 'hello', value: '{"foo":"bar"}'});
    });

    it('extends the data.constants with the options.constants', function () {
        var data = { constants: { hello: 'andrew' } };
        var opts = { constants: { hello: 'world' } };
        var result = ngConstant.getConstants(data, opts);
        expect(result[0]).toEqual({ name: 'hello', value: '"world"' });
    });

    it('uses the data object if data.constants is not available', function () {
        var data = { hello: 'andrew' };
        var result = ngConstant.getConstants(data);
        expect(result[0]).toEqual({ name: 'hello', value: '"andrew"' });
    });

    it('accepts a JSON string as constants from the options', function () {
        var opts = { constants: JSON.stringify({ hello: 'world' }) };
        var result = ngConstant.getConstants({}, opts);
        expect(result[0]).toEqual({ name: 'hello', value: '"world"' });
    });

    it('stringifies the value with the given option.space', function () {
        var data = { constants: { hello: { foo: 'bar' } } };
        var result = ngConstant.getConstants(data, { space: '' });
        expect(result[0].value).toEqual('{"foo":"bar"}');
        result = ngConstant.getConstants(data, { space: ' ' });
        expect(result[0].value).toEqual('{\n "foo": "bar"\n}');
    });
});

describe('ngConstant.getFilePath', function() {
    it('returns the file path from the src plugin when option dest is undefined', function() {
        expect(ngConstant.getFilePath('/foo/bar/config.json', {})).toBe('/foo/bar/config.js');
    });

    it('returns the file path defined by the dest option', function() {
        expect(ngConstant.getFilePath('/foo/bar/foo.js', { dest: 'foo.js' })).toBe('/foo/bar/foo.js');
    });
});
