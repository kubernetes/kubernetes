/* jshint mocha: true */
/* eslint-env node, mocha */

/**
 * Module dependencies.
 */

var ejs = require('..');
var fs = require('fs');
var read = fs.readFileSync;
var assert = require('assert');
var path = require('path');
var LRU = require('lru-cache');

try {
  fs.mkdirSync(__dirname + '/tmp');
} catch (ex) {
  if (ex.code !== 'EEXIST') {
    throw ex;
  }
}

// From https://gist.github.com/pguillory/729616
function hook_stdio(stream, callback) {
  var old_write = stream.write;

  stream.write = (function() {
    return function(string, encoding, fd) {
      callback(string, encoding, fd);
    };
  })(stream.write);

  return function() {
    stream.write = old_write;
  };
}

/**
 * Load fixture `name`.
 */

function fixture(name) {
  return read('test/fixtures/' + name, 'utf8');
}

/**
 * User fixtures.
 */

var users = [];
users.push({name: 'geddy'});
users.push({name: 'neil'});
users.push({name: 'alex'});

suite('ejs.compile(str, options)', function () {
  test('compile to a function', function () {
    var fn = ejs.compile('<p>yay</p>');
    assert.equal(fn(), '<p>yay</p>');
  });

  test('empty input works', function () {
    var fn = ejs.compile('');
    assert.equal(fn(), '');
  });

  test('throw if there are syntax errors', function () {
    try {
      ejs.compile(fixture('fail.ejs'));
    }
    catch (err) {
      assert.ok(err.message.indexOf('compiling ejs') > -1);

      try {
        ejs.compile(fixture('fail.ejs'), {filename: 'fail.ejs'});
      }
      catch (err) {
        assert.ok(err.message.indexOf('fail.ejs') > -1);
        return;
      }
    }
    throw new Error('no error reported when there should be');
  });

  test('allow customizing delimiter local var', function () {
    var fn;
    fn = ejs.compile('<p><?= name ?></p>', {delimiter: '?'});
    assert.equal(fn({name: 'geddy'}), '<p>geddy</p>');

    fn = ejs.compile('<p><:= name :></p>', {delimiter: ':'});
    assert.equal(fn({name: 'geddy'}), '<p>geddy</p>');

    fn = ejs.compile('<p><$= name $></p>', {delimiter: '$'});
    assert.equal(fn({name: 'geddy'}), '<p>geddy</p>');
  });

  test('default to using ejs.delimiter', function () {
    var fn;
    ejs.delimiter = '&';
    fn = ejs.compile('<p><&= name &></p>');
    assert.equal(fn({name: 'geddy'}), '<p>geddy</p>');

    fn = ejs.compile('<p><|= name |></p>', {delimiter: '|'});
    assert.equal(fn({name: 'geddy'}), '<p>geddy</p>');
    delete ejs.delimiter;
  });

  test('support custom escape function', function () {
    var customEscape;
    var fn;
    customEscape = function customEscape(str) {
      return !str ? '' : str.toUpperCase();
    };
    fn = ejs.compile('HELLO <%= name %>', {escape: customEscape});
    assert.equal(fn({name: 'world'}), 'HELLO WORLD');
  });

  test('strict mode works', function () {
    assert.equal(ejs.render(fixture('strict.ejs'), {}, {strict: true}), 'true');
  });

});

suite('client mode', function () {

  test('have a working client option', function () {
    var fn;
    var str;
    var preFn;
    fn = ejs.compile('<p><%= foo %></p>', {client: true});
    str = fn.toString();
    if (!process.env.running_under_istanbul) {
      eval('var preFn = ' + str);
      assert.equal(preFn({foo: 'bar'}), '<p>bar</p>');
    }
  });

  test('support client mode without locals', function () {
    var fn;
    var str;
    var preFn;
    fn = ejs.compile('<p><%= "foo" %></p>', {client: true});
    str = fn.toString();
    if (!process.env.running_under_istanbul) {
      eval('var preFn = ' + str);
      assert.equal(preFn(), '<p>foo</p>');
    }
  });

  test('not include rethrow() in client mode if compileDebug is false', function () {
    var fn = ejs.compile('<p><%= "foo" %></p>', {
      client: true,
      compileDebug: false
    });
    // There could be a `rethrow` in the function declaration
    assert((fn.toString().match(/rethrow/g) || []).length <= 1);
  });

  test('support custom escape function in client mode', function () {
    var customEscape;
    var fn;
    var str;
    customEscape = function customEscape(str) {
      return !str ? '' : str.toUpperCase();
    };
    fn = ejs.compile('HELLO <%= name %>', {escape: customEscape, client: true});
    str = fn.toString();
    if (!process.env.running_under_istanbul) {
      eval('var preFn = ' + str);
      assert.equal(preFn({name: 'world'}), 'HELLO WORLD'); // eslint-disable-line no-undef
    }
  });

  test('escape filename in errors in client mode', function () {
    assert.throws(function () {
      var fn = ejs.compile('<% throw new Error("whoops"); %>', {client: true, filename: '<script>'});
      fn();
    }, /Error: &lt;script&gt;/);
  });
});

/* Old API -- remove when this shim goes away */
suite('ejs.render(str, dataAndOpts)', function () {
  test('render the template with data/opts passed together', function () {
    assert.equal(ejs.render('<p><?= foo ?></p>', {foo: 'yay', delimiter: '?'}),
        '<p>yay</p>');
  });

  test('disallow unsafe opts passed along in data', function () {
    assert.equal(ejs.render('<p><?= locals.foo ?></p>',
        // localsName should not get reset because it's blacklisted
        {_with: false, foo: 'yay', delimiter: '?', localsName: '_'}),
        '<p>yay</p>');
  });
});

suite('ejs.render(str, data, opts)', function () {
  test('render the template', function () {
    assert.equal(ejs.render('<p>yay</p>'), '<p>yay</p>');
  });

  test('empty input works', function () {
    assert.equal(ejs.render(''), '');
  });

  test('undefined renders nothing escaped', function () {
    assert.equal(ejs.render('<%= undefined %>'), '');
  });

  test('undefined renders nothing raw', function () {
    assert.equal(ejs.render('<%- undefined %>'), '');
  });

  test('null renders nothing escaped', function () {
    assert.equal(ejs.render('<%= null %>'), '');
  });

  test('null renders nothing raw', function () {
    assert.equal(ejs.render('<%- null %>'), '');
  });

  test('zero-value data item renders something escaped', function () {
    assert.equal(ejs.render('<%= 0 %>'), '0');
  });

  test('zero-value data object renders something raw', function () {
    assert.equal(ejs.render('<%- 0 %>'), '0');
  });

  test('accept locals', function () {
    assert.equal(ejs.render('<p><%= name %></p>', {name: 'geddy'}),
        '<p>geddy</p>');
  });

  test('accept locals without using with() {}', function () {
    assert.equal(ejs.render('<p><%= locals.name %></p>', {name: 'geddy'},
                            {_with: false}),
        '<p>geddy</p>');
    assert.throws(function() {
      ejs.render('<p><%= name %></p>', {name: 'geddy'},
                 {_with: false});
    }, /name is not defined/);
  });

  test('accept custom name for locals', function () {
    ejs.localsName = 'it';
    assert.equal(ejs.render('<p><%= it.name %></p>', {name: 'geddy'},
                            {_with: false}),
        '<p>geddy</p>');
    assert.throws(function() {
      ejs.render('<p><%= name %></p>', {name: 'geddy'},
                 {_with: false});
    }, /name is not defined/);
    ejs.localsName = 'locals';
  });

  test('support caching', function () {
    var file = __dirname + '/tmp/render.ejs';
    var options = {cache: true, filename: file};
    var out = ejs.render('<p>Old</p>', {}, options);
    var expected = '<p>Old</p>';
    assert.equal(out, expected);
    // Assert no change, still in cache
    out = ejs.render('<p>New</p>', {}, options);
    assert.equal(out, expected);
  });

  test('support LRU caching', function () {
    var oldCache = ejs.cache;
    var file = __dirname + '/tmp/render.ejs';
    var options = {cache: true, filename: file};
    var out;
    var expected = '<p>Old</p>';

    // Switch to LRU
    ejs.cache = LRU();

    out = ejs.render('<p>Old</p>', {}, options);
    assert.equal(out, expected);
    // Assert no change, still in cache
    out = ejs.render('<p>New</p>', {}, options);
    assert.equal(out, expected);

    // Restore system cache
    ejs.cache = oldCache;
  });

  test('opts.context', function () {
    var ctxt = {foo: 'FOO'};
    var out = ejs.render('<%= this.foo %>', {}, {context: ctxt});
    assert.equal(out, ctxt.foo);
  });
});

suite('ejs.renderFile(path, [data], [options], fn)', function () {
  test('render a file', function(done) {
    ejs.renderFile('test/fixtures/para.ejs', function(err, html) {
      if (err) {
        return done(err);
      }
      assert.equal(html, '<p>hey</p>\n');
      done();
    });
  });

  test('accept locals', function(done) {
    var data =  {name: 'fonebone'};
    var options = {delimiter: '$'};
    ejs.renderFile('test/fixtures/user.ejs', data, options, function(err, html) {
      if (err) {
        return done(err);
      }
      assert.equal(html, '<h1>fonebone</h1>\n');
      done();
    });
  });

  test('accept locals without using with() {}', function(done) {
    var data =  {name: 'fonebone'};
    var options = {delimiter: '$', _with: false};
    var doneCount = 0;
    ejs.renderFile('test/fixtures/user-no-with.ejs', data, options, function(err, html) {
      if (err) {
        if (doneCount === 2) {
          return;
        }
        doneCount = 2;
        return done(err);
      }
      assert.equal(html, '<h1>fonebone</h1>\n');
      doneCount++;
      if (doneCount === 2) {
        done();
      }
    });
    ejs.renderFile('test/fixtures/user.ejs', data, options, function(err) {
      if (!err) {
        if (doneCount === 2) {
          return;
        }
        doneCount = 2;
        return done(new Error('error not thrown'));
      }
      doneCount++;
      if (doneCount === 2) {
        done();
      }
    });
  });

  test('not catch err thrown by callback', function(done) {
    var data =  {name: 'fonebone'};
    var options = {delimiter: '$'};
    var counter = 0;

    var d = require('domain').create();
    d.on('error', function (err) {
      assert.equal(counter, 1);
      assert.equal(err.message, 'Exception in callback');
      done();
    });
    d.run(function () {
      // process.nextTick() needed to work around mochajs/mocha#513
      //
      // tl;dr: mocha doesn't support synchronous exception throwing in
      // domains. Have to make it async. Ticket closed because: "domains are
      // deprecated :D"
      process.nextTick(function () {
        ejs.renderFile('test/fixtures/user.ejs', data, options, function(err) {
          counter++;
          if (err) {
            assert.notEqual(err.message, 'Exception in callback');
            return done(err);
          }
          throw new Error('Exception in callback');
        });
      });
    });
  });

  test('support caching', function (done) {
    var expected = '<p>Old</p>';
    var file = __dirname + '/tmp/renderFile.ejs';
    var options = {cache: true};
    fs.writeFileSync(file, '<p>Old</p>');

    ejs.renderFile(file, {}, options, function (err, out) {
      if (err) {
        done(err);
      }
      fs.writeFileSync(file, '<p>New</p>');
      assert.equal(out, expected);

      ejs.renderFile(file, {}, options, function (err, out) {
        if (err) {
          done(err);
        }
        // Assert no change, still in cache
        assert.equal(out, expected);
        done();
      });
    });
  });

  test('opts.context', function (done) {
    var ctxt = {foo: 'FOO'};
    ejs.renderFile('test/fixtures/with-context.ejs', {}, {context: ctxt}, function(err, html) {
      if (err) {
        return done(err);
      }
      assert.equal(html, ctxt.foo + '\n');
      done();
    });

  });
});

suite('cache specific', function () {
  test('`clearCache` work properly', function () {
    var expected = '<p>Old</p>';
    var file = __dirname + '/tmp/clearCache.ejs';
    var options = {cache: true, filename: file};
    var out = ejs.render('<p>Old</p>', {}, options);
    assert.equal(out, expected);

    ejs.clearCache();

    expected = '<p>New</p>';
    out = ejs.render('<p>New</p>', {}, options);
    assert.equal(out, expected);
  });

  test('`clearCache` work properly, LRU', function () {
    var expected = '<p>Old</p>';
    var oldCache = ejs.cache;
    var file = __dirname + '/tmp/clearCache.ejs';
    var options = {cache: true, filename: file};
    var out;

    ejs.cache = LRU();

    out = ejs.render('<p>Old</p>', {}, options);
    assert.equal(out, expected);
    ejs.clearCache();
    expected = '<p>New</p>';
    out = ejs.render('<p>New</p>', {}, options);
    assert.equal(out, expected);

    ejs.cache = oldCache;
  });

  test('LRU with cache-size 1', function () {
    var oldCache = ejs.cache;
    var options;
    var out;
    var expected;
    var file;

    ejs.cache = LRU(1);

    file = __dirname + '/tmp/render1.ejs';
    options = {cache: true, filename: file};
    out = ejs.render('<p>File1</p>', {}, options);
    expected = '<p>File1</p>';
    assert.equal(out, expected);

    // Same filename, different template, but output
    // should be the same because cache
    file = __dirname + '/tmp/render1.ejs';
    options = {cache: true, filename: file};
    out = ejs.render('<p>ChangedFile1</p>', {}, options);
    expected = '<p>File1</p>';
    assert.equal(out, expected);

    // Different filiename -- output should be different,
    // and previous cache-entry should be evicted
    file = __dirname + '/tmp/render2.ejs';
    options = {cache: true, filename: file};
    out = ejs.render('<p>File2</p>', {}, options);
    expected = '<p>File2</p>';
    assert.equal(out, expected);

    // Entry with first filename should now be out of cache,
    // results should be different
    file = __dirname + '/tmp/render1.ejs';
    options = {cache: true, filename: file};
    out = ejs.render('<p>ChangedFile1</p>', {}, options);
    expected = '<p>ChangedFile1</p>';
    assert.equal(out, expected);

    ejs.cache = oldCache;
  });
});

suite('<%', function () {
  test('without semicolons', function () {
    assert.equal(ejs.render(fixture('no.semicolons.ejs')),
        fixture('no.semicolons.html'));
  });
});

suite('<%=', function () {
  test('should not throw an error with a // comment on the final line', function () {
    assert.equal(ejs.render('<%=\n// a comment\nname\n// another comment %>', {name: '&nbsp;<script>'}),
      '&amp;nbsp;&lt;script&gt;');
  });

  test('escape &amp;<script>', function () {
    assert.equal(ejs.render('<%= name %>', {name: '&nbsp;<script>'}),
        '&amp;nbsp;&lt;script&gt;');
  });

  test('should escape \'', function () {
    assert.equal(ejs.render('<%= name %>', {name: 'The Jones\'s'}),
      'The Jones&#39;s');
  });

  test('should escape &foo_bar;', function () {
    assert.equal(ejs.render('<%= name %>', {name: '&foo_bar;'}),
      '&amp;foo_bar;');
  });

  test('should accept custom function', function() {

    var customEscape = function customEscape(str) {
      return !str ? '' : str.toUpperCase();
    };

    assert.equal(
      ejs.render('<%= name %>', {name: 'The Jones\'s'}, {escape: customEscape}),
      'THE JONES\'S'
    );
  });
});

suite('<%-', function () {
  test('should not throw an error with a // comment on the final line', function () {
    assert.equal(ejs.render('<%-\n// a comment\nname\n// another comment %>', {name: '&nbsp;<script>'}),
        '&nbsp;<script>');
  });

  test('not escape', function () {
    assert.equal(ejs.render('<%- name %>', {name: '<script>'}),
        '<script>');
  });

  test('terminate gracefully if no close tag is found', function () {
    try {
      ejs.compile('<h1>oops</h1><%- name ->');
      throw new Error('Expected parse failure');
    }
    catch (err) {
      assert.ok(err.message.indexOf('Could not find matching close tag for') > -1);
    }
  });
});

suite('%>', function () {
  test('produce newlines', function () {
    assert.equal(ejs.render(fixture('newlines.ejs'), {users: users}),
      fixture('newlines.html'));
  });
  test('works with `-%>` interspersed', function () {
    assert.equal(ejs.render(fixture('newlines.mixed.ejs'), {users: users}),
      fixture('newlines.mixed.html'));
  });
  test('consecutive tags work', function () {
    assert.equal(ejs.render(fixture('consecutive-tags.ejs')),
      fixture('consecutive-tags.html'));
  });
});

suite('-%>', function () {
  test('not produce newlines', function () {
    assert.equal(ejs.render(fixture('no.newlines.ejs'), {users: users}),
      fixture('no.newlines.html'));
  });
  test('stack traces work', function () {
    try {
      ejs.render(fixture('no.newlines.error.ejs'));
    }
    catch (e) {
      if (e.message.indexOf('>> 4| <%= qdata %>') > -1) {
        return;
      }
      throw e;
    }
    throw new Error('Expected ReferenceError');
  });

  test('works with unix style', function () {
    var content = '<ul><% -%>\n'
    + '<% users.forEach(function(user){ -%>\n'
    + '<li><%= user.name -%></li>\n'
    + '<% }) -%>\n'
    + '</ul><% -%>\n';

    var expectedResult = '<ul><li>geddy</li>\n<li>neil</li>\n<li>alex</li>\n</ul>';
    var fn;
    fn = ejs.compile(content);
    assert.equal(fn({users: users}),
      expectedResult);
  });

  test('works with windows style', function () {
    var content = '<ul><% -%>\r\n'
    + '<% users.forEach(function(user){ -%>\r\n'
    + '<li><%= user.name -%></li>\r\n'
    + '<% }) -%>\r\n'
    + '</ul><% -%>\r\n';

    var expectedResult = '<ul><li>geddy</li>\r\n<li>neil</li>\r\n<li>alex</li>\r\n</ul>';
    var fn;
    fn = ejs.compile(content);
    assert.equal(fn({users: users}),
      expectedResult);
  });
});

suite('<%%', function () {
  test('produce literals', function () {
    assert.equal(ejs.render('<%%- "foo" %>'),
      '<%- "foo" %>');
  });
  test('work without an end tag', function () {
    assert.equal(ejs.render('<%%'), '<%');
    assert.equal(ejs.render(fixture('literal.ejs'), {}, {delimiter: ' '}),
      fixture('literal.html'));
  });
});

suite('%%>', function () {
  test('produce literal', function () {
    assert.equal(ejs.render('%%>'),
        '%>');
    assert.equal(ejs.render('  >', {}, {delimiter: ' '}),
        ' >');
  });
});

suite('<%_ and _%>', function () {
  test('slurps spaces and tabs', function () {
    assert.equal(ejs.render(fixture('space-and-tab-slurp.ejs'), {users: users}),
      fixture('space-and-tab-slurp.html'));
  });
});

suite('single quotes', function () {
  test('not mess up the constructed function', function () {
    assert.equal(ejs.render(fixture('single-quote.ejs')),
      fixture('single-quote.html'));
  });
});

suite('double quotes', function () {
  test('not mess up the constructed function', function () {
    assert.equal(ejs.render(fixture('double-quote.ejs')),
      fixture('double-quote.html'));
  });
});

suite('backslashes', function () {
  test('escape', function () {
    assert.equal(ejs.render(fixture('backslash.ejs')),
      fixture('backslash.html'));
  });
});

suite('messed up whitespace', function () {
  test('work', function () {
    assert.equal(ejs.render(fixture('messed.ejs'), {users: users}),
      fixture('messed.html'));
  });
});

suite('exceptions', function () {
  test('produce useful stack traces', function () {
    try {
      ejs.render(fixture('error.ejs'), {}, {filename: 'error.ejs'});
    }
    catch (err) {
      assert.equal(err.path, 'error.ejs');
      assert.equal(err.stack.split('\n').slice(0, 8).join('\n'), fixture('error.out'));
      return;
    }
    throw new Error('no error reported when there should be');
  });

  test('not include fancy stack info if compileDebug is false', function () {
    try {
      ejs.render(fixture('error.ejs'), {}, {
        filename: 'error.ejs',
        compileDebug: false
      });
    }
    catch (err) {
      assert.ok(!err.path);
      assert.notEqual(err.stack.split('\n').slice(0, 8).join('\n'), fixture('error.out'));
      return;
    }
    throw new Error('no error reported when there should be');
  });

  var unhook = null;
  test('log JS source when debug is set', function (done) {
    var out = '';
    var needToExit = false;
    unhook = hook_stdio(process.stdout, function (str) {
      out += str;
      if (needToExit) {
        return;
      }
      if (out.indexOf('__output')) {
        needToExit = true;
        unhook();
        unhook = null;
        return done();
      }
    });
    ejs.render(fixture('hello-world.ejs'), {}, {debug: true});
  });

  test('escape filename in errors', function () {
    assert.throws(function () {
      ejs.render('<% throw new Error("whoops"); %>', {}, {filename: '<script>'});
    }, /Error: &lt;script&gt;/);
  });

  test('filename in errors uses custom escape', function () {
    assert.throws(function () {
      ejs.render('<% throw new Error("whoops"); %>', {}, {
        filename: '<script>',
        escape: function () { return 'zooby'; }
      });
    }, /Error: zooby/);
  });

  teardown(function() {
    if (!unhook) {
      return;
    }
    unhook();
    unhook = null;
  });
});

suite('rmWhitespace', function () {
  test('works', function () {
    assert.equal(ejs.render(fixture('rmWhitespace.ejs'), {}, {rmWhitespace: true}),
        fixture('rmWhitespace.html'));
  });
});

suite('include()', function () {
  test('include ejs', function () {
    var file = 'test/fixtures/include-simple.ejs';
    assert.equal(ejs.render(fixture('include-simple.ejs'), {}, {filename: file}),
        fixture('include-simple.html'));
  });

  test('include ejs fails without `filename`', function () {
    try {
      ejs.render(fixture('include-simple.ejs'));
    }
    catch (err) {
      assert.ok(err.message.indexOf('requires the \'filename\' option') > -1);
      return;
    }
    throw new Error('expected inclusion error');
  });

  test('strips BOM', function () {
    assert.equal(
      ejs.render('<%- include("fixtures/includes/bom.ejs") %>',
        {}, {filename: path.join(__dirname, 'f.ejs')}),
      '<p>This is a file with BOM.</p>\n');
  });

  test('include ejs with locals', function () {
    var file = 'test/fixtures/include.ejs';
    assert.equal(ejs.render(fixture('include.ejs'), {pets: users}, {filename: file, delimiter: '@'}),
        fixture('include.html'));
  });

  test('include ejs with absolute path and locals', function () {
    var file = 'test/fixtures/include-abspath.ejs';
    assert.equal(ejs.render(fixture('include-abspath.ejs'),
      {dir: path.join(__dirname, 'fixtures'), pets: users, path: path},
      {filename: file, delimiter: '@'}),
        fixture('include.html'));
  });

  test('include ejs with set root path', function () {
    var file = 'test/fixtures/include-root.ejs';
    var viewsPath = path.join(__dirname, 'fixtures');
    assert.equal(ejs.render(fixture('include-root.ejs'), {pets: users}, {filename: file, delimiter: '@',root:viewsPath}),
      fixture('include.html'));
  });

  test('work when nested', function () {
    var file = 'test/fixtures/menu.ejs';
    assert.equal(ejs.render(fixture('menu.ejs'), {pets: users}, {filename: file}),
        fixture('menu.html'));
  });

  test('work with a variable path', function () {
    var file = 'test/fixtures/menu_var.ejs';
    var includePath = 'includes/menu-item';
    assert.equal(ejs.render(fixture('menu.ejs'), {pets: users, varPath:  includePath}, {filename: file}),
      fixture('menu.html'));
  });

  test('include arbitrary files as-is', function () {
    var file = 'test/fixtures/include.css.ejs';
    assert.equal(ejs.render(fixture('include.css.ejs'), {pets: users}, {filename: file}),
        fixture('include.css.html'));
  });

  test('pass compileDebug to include', function () {
    var file = 'test/fixtures/include.ejs';
    var fn;
    fn = ejs.compile(fixture('include.ejs'), {
      filename: file,
      delimiter: '@',
      compileDebug: false
    });
    try {
      // Render without a required variable reference
      fn({foo: 'asdf'});
    }
    catch(e) {
      assert.equal(e.message, 'pets is not defined');
      assert.ok(!e.path);
      return;
    }
    throw new Error('no error reported when there should be');
  });

  test('is dynamic', function () {
    fs.writeFileSync(__dirname + '/tmp/include.ejs', '<p>Old</p>');
    var file = 'test/fixtures/include_cache.ejs';
    var options = {filename: file};
    var out = ejs.compile(fixture('include_cache.ejs'), options);
    assert.equal(out(), '<p>Old</p>\n');

    fs.writeFileSync(__dirname + '/tmp/include.ejs', '<p>New</p>');
    assert.equal(out(), '<p>New</p>\n');
  });

  test('support caching', function () {
    fs.writeFileSync(__dirname + '/tmp/include.ejs', '<p>Old</p>');
    var file = 'test/fixtures/include_cache.ejs';
    var options = {cache: true, filename: file};
    var out = ejs.render(fixture('include_cache.ejs'), {}, options);
    var expected = fixture('include_cache.html');
    assert.equal(out, expected);
    out = ejs.render(fixture('include_cache.ejs'), {}, options);
    // No change, still in cache
    assert.equal(out, expected);
    fs.writeFileSync(__dirname + '/tmp/include.ejs', '<p>New</p>');
    out = ejs.render(fixture('include_cache.ejs'), {}, options);
    assert.equal(out, expected);
  });

});

suite('preprocessor include', function () {
  test('work', function () {
    var file = 'test/fixtures/include_preprocessor.ejs';
    assert.equal(ejs.render(fixture('include_preprocessor.ejs'), {pets: users}, {filename: file, delimiter: '@'}),
        fixture('include_preprocessor.html'));
  });

  test('no false positives', function () {
    assert.equal(ejs.render('<% %> include foo <% %>'), ' include foo ');
  });

  test('fails without `filename`', function () {
    try {
      ejs.render(fixture('include_preprocessor.ejs'), {pets: users}, {delimiter: '@'});
    }
    catch (err) {
      assert.ok(err.message.indexOf('requires the \'filename\' option') > -1);
      return;
    }
    throw new Error('expected inclusion error');
  });

  test('strips BOM', function () {
    assert.equal(
      ejs.render('<% include fixtures/includes/bom.ejs %>',
        {}, {filename: path.join(__dirname, 'f.ejs')}),
      '<p>This is a file with BOM.</p>\n');
  });

  test('work when nested', function () {
    var file = 'test/fixtures/menu_preprocessor.ejs';
    assert.equal(ejs.render(fixture('menu_preprocessor.ejs'), {pets: users}, {filename: file}),
        fixture('menu_preprocessor.html'));
  });

  test('tracks dependency correctly', function () {
    var file = 'test/fixtures/menu_preprocessor.ejs';
    var fn = ejs.compile(fixture('menu_preprocessor.ejs'), {filename: file});
    assert(fn.dependencies.length);
  });

  test('include arbitrary files as-is', function () {
    var file = 'test/fixtures/include_preprocessor.css.ejs';
    assert.equal(ejs.render(fixture('include_preprocessor.css.ejs'), {pets: users}, {filename: file}),
        fixture('include_preprocessor.css.html'));
  });

  test('pass compileDebug to include', function () {
    var file = 'test/fixtures/include_preprocessor.ejs';
    var fn;
    fn = ejs.compile(fixture('include_preprocessor.ejs'), {
      filename: file,
      delimiter: '@',
      compileDebug: false
    });
    try {
      // Render without a required variable reference
      fn({foo: 'asdf'});
    }
    catch(e) {
      assert.equal(e.message, 'pets is not defined');
      assert.ok(!e.path);
      return;
    }
    throw new Error('no error reported when there should be');
  });

  test('is static', function () {
    fs.writeFileSync(__dirname + '/tmp/include_preprocessor.ejs', '<p>Old</p>');
    var file = 'test/fixtures/include_preprocessor_cache.ejs';
    var options = {filename: file};
    var out = ejs.compile(fixture('include_preprocessor_cache.ejs'), options);
    assert.equal(out(), '<p>Old</p>\n');

    fs.writeFileSync(__dirname + '/tmp/include_preprocessor.ejs', '<p>New</p>');
    assert.equal(out(), '<p>Old</p>\n');
  });

  test('support caching', function () {
    fs.writeFileSync(__dirname + '/tmp/include_preprocessor.ejs', '<p>Old</p>');
    var file = 'test/fixtures/include_preprocessor_cache.ejs';
    var options = {cache: true, filename: file};
    var out = ejs.render(fixture('include_preprocessor_cache.ejs'), {}, options);
    var expected = fixture('include_preprocessor_cache.html');
    assert.equal(out, expected);
    fs.writeFileSync(__dirname + '/tmp/include_preprocessor.ejs', '<p>New</p>');
    out = ejs.render(fixture('include_preprocessor_cache.ejs'), {}, options);
    assert.equal(out, expected);
  });

  test('whitespace slurp and rmWhitespace work', function() {
    var file = 'test/fixtures/include_preprocessor_line_slurp.ejs';
    var template = fixture('include_preprocessor_line_slurp.ejs');
    var expected = fixture('include_preprocessor_line_slurp.html');
    var options = {rmWhitespace: true, filename: file};
    assert.equal(ejs.render(template, {}, options),
        expected);
  });

});

suite('comments', function () {
  test('fully render with comments removed', function () {
    assert.equal(ejs.render(fixture('comments.ejs')),
        fixture('comments.html'));
  });
});

suite('require', function () {

  // Only works with inline/preprocessor includes
  test('allow ejs templates to be required as node modules', function () {
    var file = 'test/fixtures/include_preprocessor.ejs';
    var template = require(__dirname + '/fixtures/menu_preprocessor.ejs');
    if (!process.env.running_under_istanbul) {
      assert.equal(template({filename: file, pets: users}),
          fixture('menu_preprocessor.html'));
    }
  });
});

suite('test fileloader', function () {

  var myFileLoad = function (filePath) {
    return 'myFileLoad: ' + fs.readFileSync(filePath);
  };

  test('test custom fileload', function (done) {
    ejs.fileLoader = myFileLoad;
    ejs.renderFile('test/fixtures/para.ejs', function(err, html) {
      if (err) {
        return done(err);
      }
      assert.equal(html, 'myFileLoad: <p>hey</p>\n');
      done();
    });

  });
});

suite('examples', function () {
  function noop () {}
  fs.readdirSync('examples').forEach(function (f) {
    if (!/\.js$/.test(f)) {
      return;
    }
    suite(f, function () {
      test('doesn\'t throw any errors', function () {
        var stderr = hook_stdio(process.stderr, noop);
        var stdout = hook_stdio(process.stdout, noop);
        try {
          require('../examples/' + f);
        }
        catch (ex) {
          stdout();
          stderr();
          throw ex;
        }
        stdout();
        stderr();
      });
    });
  });
});

suite('meta information', function () {
  test('has a version', function () {
    assert.strictEqual(ejs.VERSION, require('../package.json').version);
  });

  test('had a name', function () {
    assert.strictEqual(ejs.name, 'ejs');
  });
});
