var should = require('chai').should(),
    yargs = require('../'),
    path = require('path');

describe('parse', function () {

    it('should pass when specifying a "short boolean"', function () {
        var parse = yargs.parse([ '-b' ]);
        parse.should.have.property('b').to.be.ok.and.be.a('boolean');
        parse.should.have.property('_').with.length(0);
    });

    it('should pass when specifying a "long boolean"', function () {
        var parse = yargs.parse(['--bool']);
        parse.should.have.property('bool', true);
        parse.should.have.property('_').with.length(0);
    });

    it('should place bare options in the _ array', function () {
        var parse = yargs.parse(['foo', 'bar', 'baz']);
        parse.should.have.property('_').and.deep.equal(['foo','bar','baz']);
    });

    it('should expand grouped short options to a hash with a key for each', function () {
        var parse = yargs.parse(['-cats']);
        parse.should.have.property('c', true);
        parse.should.have.property('a', true);
        parse.should.have.property('t', true);
        parse.should.have.property('s', true);
        parse.should.have.property('_').with.length(0);
    });

    it('should set the value of the final option in a group to the next supplied value', function () {
        var parse = yargs.parse(['-cats', 'meow']);
        parse.should.have.property('c', true);
        parse.should.have.property('a', true);
        parse.should.have.property('t', true);
        parse.should.have.property('s', 'meow');
        parse.should.have.property('_').with.length(0);
    });

    it('should set the value of a single short option to the next supplied value', function () {
        var parse = yargs.parse(['-h', 'localhost']);
        parse.should.have.property('h', 'localhost');
        parse.should.have.property('_').with.length(0);
    });

    it('should set the value of multiple single short options to the next supplied values relative to each', function () {
        var parse = yargs.parse(['-h', 'localhost', '-p', '555']);
        parse.should.have.property('h', 'localhost');
        parse.should.have.property('p', 555);
        parse.should.have.property('_').with.length(0);
    });

    it('should set the value of a single long option to the next supplied value', function () {
        var parse = yargs.parse(['--pow', 'xixxle']);
        parse.should.have.property('pow', 'xixxle');
        parse.should.have.property('_').with.length(0);
    });

    it('should set the value of a single long option if an = was used', function () {
        var parse = yargs.parse(['--pow=xixxle']);
        parse.should.have.property('pow', 'xixxle');
        parse.should.have.property('_').with.length(0);
    });

    it('should set the value of multiple long options to the next supplied values relative to each', function () {
        var parse = yargs.parse(['--host', 'localhost', '--port', '555']);
        parse.should.have.property('host', 'localhost');
        parse.should.have.property('port', 555);
        parse.should.have.property('_').with.length(0);
    });

    it('should set the value of multiple long options if = signs were used', function () {
        var parse = yargs.parse(['--host=localhost', '--port=555']);
        parse.should.have.property('host', 'localhost');
        parse.should.have.property('port', 555);
        parse.should.have.property('_').with.length(0);
    });

    it('should still set values appropriately if a mix of short, long, and grouped short options are specified', function () {
        var parse = yargs.parse(['-h', 'localhost', '-fp', '555', 'script.js']);
        parse.should.have.property('f', true);
        parse.should.have.property('p', 555);
        parse.should.have.property('h', 'localhost');
        parse.should.have.property('_').and.deep.equal(['script.js']);
    });

    it('should still set values appropriately if a mix of short and long options are specified', function () {
        var parse = yargs.parse(['-h', 'localhost', '--port', '555']);
        parse.should.have.property('h', 'localhost');
        parse.should.have.property('port', 555);
        parse.should.have.property('_').with.length(0);
    });

    it('should explicitly set a boolean option to false if preceeded by "--no-"', function () {
        var parse = yargs.parse(['--no-moo']);
        parse.should.have.property('moo', false);
        parse.should.have.property('_').with.length(0);
    });

    it('should group values into an array if the same option is specified multiple times', function () {
        var parse = yargs.parse(['-v', 'a', '-v', 'b', '-v', 'c' ]);
        parse.should.have.property('v').and.deep.equal(['a','b','c']);
        parse.should.have.property('_').with.length(0);
    });

    it('should still set values appropriately if we supply a comprehensive list of various types of options', function () {
        var parse = yargs.parse([
            '--name=meowmers', 'bare', '-cats', 'woo',
            '-h', 'awesome', '--multi=quux',
            '--key', 'value',
            '-b', '--bool', '--no-meep', '--multi=baz',
            '--', '--not-a-flag', 'eek'
        ]);
        parse.should.have.property('c', true);
        parse.should.have.property('a', true);
        parse.should.have.property('t', true);
        parse.should.have.property('s', 'woo');
        parse.should.have.property('h', 'awesome');
        parse.should.have.property('b', true);
        parse.should.have.property('bool', true);
        parse.should.have.property('key', 'value');
        parse.should.have.property('multi').and.deep.equal(['quux', 'baz']);
        parse.should.have.property('meep', false);
        parse.should.have.property('name', 'meowmers');
        parse.should.have.property('_').and.deep.equal(['bare', '--not-a-flag', 'eek']);
    });

    it('should parse numbers appropriately', function () {
        var argv = yargs.parse([
            '-x', '1234',
            '-y', '5.67',
            '-z', '1e7',
            '-w', '10f',
            '--hex', '0xdeadbeef',
            '789',
        ]);
        argv.should.have.property('x', 1234).and.be.a('number');
        argv.should.have.property('y', 5.67).and.be.a('number');
        argv.should.have.property('z', 1e7).and.be.a('number');
        argv.should.have.property('w', '10f').and.be.a('string');
        argv.should.have.property('hex', 0xdeadbeef).and.be.a('number');
        argv.should.have.property('_').and.deep.equal([789]);
        argv._[0].should.be.a('number');
    });

    it('should not set the next value as the value of a short option if that option is explicitly defined as a boolean', function () {
        var parse = yargs([ '-t', 'moo' ]).boolean(['t']).argv;
        parse.should.have.property('t', true).and.be.a('boolean');
        parse.should.have.property('_').and.deep.equal(['moo']);
    });

    it('should set boolean options values if the next value is "true" or "false"', function () {
        var parse = yargs(['--verbose', 'false', 'moo', '-t', 'true'])
            .boolean(['t', 'verbose']).default('verbose', true).argv;
        parse.should.have.property('verbose', false).and.be.a('boolean');
        parse.should.have.property('t', true).and.be.a('boolean');
        parse.should.have.property('_').and.deep.equal(['moo']);
    });

    it('should set boolean options to false by default', function () {
        var parse = yargs(['moo'])
            .boolean(['t', 'verbose'])
            .default('verbose', false)
            .default('t', false).argv;
        parse.should.have.property('verbose', false).and.be.a('boolean');
        parse.should.have.property('t', false).and.be.a('boolean');
        parse.should.have.property('_').and.deep.equal(['moo']);
    });

    it('should allow defining options as boolean in groups', function () {
        var parse = yargs([ '-x', '-z', 'one', 'two', 'three' ])
            .boolean(['x','y','z']).argv;
        parse.should.have.property('x', true).and.be.a('boolean');
        parse.should.have.property('y', false).and.be.a('boolean');
        parse.should.have.property('z', true).and.be.a('boolean');
        parse.should.have.property('_').and.deep.equal(['one','two','three']);
    });

    it('should preserve newlines in option values' , function () {
        var args = yargs.parse(['-s', "X\nX"]);
        args.should.have.property('_').with.length(0);
        args.should.have.property('s', 'X\nX');
        // reproduce in bash:
        // VALUE="new
        // line"
        // node program.js --s="$VALUE"
        args = yargs.parse(["--s=X\nX"]);
        args.should.have.property('_').with.length(0);
        args.should.have.property('s', 'X\nX');
    });

    it('should not convert numbers to type number if explicitly defined as strings' , function () {
        var s = yargs([ '-s', '0001234' ]).string('s').argv.s;
        s.should.be.a('string').and.equal('0001234');
        var x = yargs([ '-x', '56' ]).string('x').argv.x;
        x.should.be.a('string').and.equal('56');
    });

    it('should leave all non-hyphenated values as strings if _ is defined as a string', function () {
        var s = yargs([ '  ', '  ' ]).string('_').argv._;
        s.should.have.length(2);
        s[0].should.be.a('string').and.equal('  ');
        s[1].should.be.a('string').and.equal('  ');
    });

    it('should normalize redundant paths', function () {
        var a = yargs([ '-s', '/tmp/../' ]).alias('s', 'save').normalize('s').argv;
        a.should.have.property('s', '/');
        a.should.have.property('save', '/');
    });

    it('should normalize redundant paths when a value is later assigned', function () {
        var a = yargs(['-s']).normalize('s').argv;
        a.should.have.property('s', true);
        a.s = '/path/to/new/dir/../../';
        a.s.should.equal('/path/to/');
    });

    it('should assign data after forward slash to the option before the slash', function () {
        var parse = yargs.parse(['-I/foo/bar/baz']);
        parse.should.have.property('_').with.length(0);
        parse.should.have.property('I', '/foo/bar/baz');
        parse = yargs.parse(['-xyz/foo/bar/baz']);
        parse.should.have.property('x', true);
        parse.should.have.property('y', true);
        parse.should.have.property('z', '/foo/bar/baz');
        parse.should.have.property('_').with.length(0);
    });

    it('should set alias value to the same value as the full option', function () {
        var argv = yargs([ '-f', '11', '--zoom', '55' ])
            .alias('z', 'zoom')
            .argv;
        argv.should.have.property('zoom', 55);
        argv.should.have.property('z', 55);
        argv.should.have.property('f', 11);
    });

    /*
     *it('should load options and values from a file when config is used', function () {
     *    var argv = yargs([ '--settings', '../test/config.json', '--foo', 'bar' ])
     *        .alias('z', 'zoom')
     *        .config('settings')
     *        .argv;
     *    argv.should.have.property('herp', 'derp');
     *    argv.should.have.property('zoom', 55);
     *    argv.should.have.property('foo').and.deep.equal(['baz','bar']);
     *});
     */

    it('should allow multiple aliases to be specified', function () {
        var argv = yargs([ '-f', '11', '--zoom', '55' ])
            .alias('z', [ 'zm', 'zoom' ])
            .argv;
        argv.should.have.property('zoom', 55);
        argv.should.have.property('z', 55);
        argv.should.have.property('zm', 55);
        argv.should.have.property('f', 11);
    });

    it('should define option as boolean and set default to true', function () {
        var argv = yargs.options({
            sometrue: {
                boolean: true,
                default: true
            }
        }).argv;
        argv.should.have.property('sometrue', true);
    });

    it('should define option as boolean and set default to false', function () {
        var argv = yargs.options({
            somefalse: {
                boolean: true,
                default: false
            }
        }).argv;
        argv.should.have.property('somefalse', false);
    });

    it('should allow object graph traversal via dot notation', function () {
        var argv = yargs([
            '--foo.bar', '3', '--foo.baz', '4',
            '--foo.quux.quibble', '5', '--foo.quux.o_O',
            '--beep.boop'
        ]).argv;
        argv.should.have.property('foo').and.deep.equal({
            bar: 3,
            baz: 4,
            quux: {
                quibble: 5,
                o_O: true
            }
        });
        argv.should.have.property('beep').and.deep.equal({ boop: true });
    });

    it('should allow booleans and aliases to be defined with chainable api', function () {
        var aliased = [ '-h', 'derp' ],
            regular = [ '--herp',  'derp' ],
            opts = {
                herp: { alias: 'h', boolean: true }
            },
            aliasedArgv = yargs(aliased).boolean('herp').alias('h', 'herp').argv,
            propertyArgv = yargs(regular).boolean('herp').alias('h', 'herp').argv;
        aliasedArgv.should.have.property('herp', true);
        aliasedArgv.should.have.property('h', true);
        aliasedArgv.should.have.property('_').and.deep.equal(['derp']);
        propertyArgv.should.have.property('herp', true);
        propertyArgv.should.have.property('h', true);
        propertyArgv.should.have.property('_').and.deep.equal(['derp']);
    });

    it('should allow booleans and aliases to be defined with options hash', function () {
        var aliased = [ '-h', 'derp' ],
            regular = [ '--herp', 'derp' ],
            opts = {
                herp: { alias: 'h', boolean: true }
            },
            aliasedArgv = yargs(aliased).options(opts).argv,
            propertyArgv = yargs(regular).options(opts).argv;
        aliasedArgv.should.have.property('herp', true);
        aliasedArgv.should.have.property('h', true);
        aliasedArgv.should.have.property('_').and.deep.equal(['derp']);
        propertyArgv.should.have.property('herp', true);
        propertyArgv.should.have.property('h', true);
        propertyArgv.should.have.property('_').and.deep.equal(['derp']);
    });

    it('should set boolean and alias using explicit true', function () {
        var aliased = [ '-h', 'true' ],
            regular = [ '--herp',  'true' ],
            opts = {
                herp: { alias: 'h', boolean: true }
            },
            aliasedArgv = yargs(aliased).boolean('h').alias('h', 'herp').argv,
            propertyArgv = yargs(regular).boolean('h').alias('h', 'herp').argv;
        aliasedArgv.should.have.property('herp', true);
        aliasedArgv.should.have.property('h', true);
        aliasedArgv.should.have.property('_').with.length(0);
    });

    // regression, see https://github.com/substack/node-optimist/issues/71
    it('should set boolean and --x=true', function() {
        var parsed = yargs(['--boool', '--other=true']).boolean('boool').argv;
        parsed.should.have.property('boool', true);
        parsed.should.have.property('other', 'true');
        parsed = yargs(['--boool', '--other=false']).boolean('boool').argv;
        parsed.should.have.property('boool', true);
        parsed.should.have.property('other', 'false');
    });

    // regression, see https://github.com/chevex/yargs/issues/63
    it('should not add the same key to argv multiple times, when creating camel-case aliases', function() {
      var yargs = require('../')(['--health-check=banana', '--second-key', 'apple', '-t=blarg'])
          .options('h', {
            alias: 'health-check',
            description: 'health check',
            default: 'apple'
          })
          .options('second-key', {
            alias: 's',
            description: 'second key',
            default: 'banana'
          })
          .options('third-key', {
            alias: 't',
            description: 'third key',
            default: 'third'
          })

      // before this fix, yargs failed parsing
      // one but not all forms of an arg.
      yargs.argv.secondKey.should.eql('apple');
      yargs.argv.s.should.eql('apple');
      yargs.argv['second-key'].should.eql('apple');

      yargs.argv.healthCheck.should.eql('banana');
      yargs.argv.h.should.eql('banana');
      yargs.argv['health-check'].should.eql('banana');

      yargs.argv.thirdKey.should.eql('blarg');
      yargs.argv.t.should.eql('blarg');
      yargs.argv['third-key'].should.eql('blarg');
    });

});
