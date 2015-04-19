var ecstatic = require('../ecstatic'),
    fs = require('fs'),
    path = require('path'),
    ent = require('ent'),
    etag = require('./etag'),
    url = require('url'),
    status = require('./status-handlers');

module.exports = function (opts, stat) {
  // opts are parsed by opts.js, defaults already applied
  var cache = opts.cache,
      root = path.resolve(opts.root),
      baseDir = opts.baseDir,
      humanReadable = opts.humanReadable,
      si = opts.si;

  return function middleware (req, res, next) {

    // Figure out the path for the file from the given url
    var parsed = url.parse(req.url),
        pathname = decodeURI(parsed.pathname),
        dir = path.normalize(
          path.join(root,
            path.relative(
              path.join('/', baseDir),
              pathname
            )
          )
        );

    fs.stat(dir, function (err, stat) {
      if (err) {
        return status[500](res, next, { error: err });
      }

      // files are the listing of dir
      fs.readdir(dir, function (err, files) {
        if (err) {
          return status[500](res, next, { error: err });
        }
        res.setHeader('content-type', 'text/html');
        res.setHeader('etag', etag(stat));
        res.setHeader('last-modified', (new Date(stat.mtime)).toUTCString());
        res.setHeader('cache-control', cache);

        sortByIsDirectory(files, function (errs, dirs, files) {

          if (errs.length > 0) {
            return status[500](res, next, { error: errs[0] });
          }

          // if it makes sense to, add a .. link
          if (path.resolve(dir, '..').slice(0, root.length) == root) {
            return fs.stat(path.join(dir, '..'), function (err, s) {
              if (err) {
                return status[500](res, next, { error: err });
              }
              dirs.unshift([ '..', s ]);
              render(dirs, files);
            });
          }
          render(dirs, files);
        });

        function sortByIsDirectory(paths, cb) {
          // take the listing file names in `dir`
          // returns directory and file array, each entry is
          // of the array a [name, stat] tuple
          var pending = paths.length,
              errs = [],
              dirs = [],
              files = [];

          if (!pending) {
            return cb(errs, dirs, files);
          }

          paths.forEach(function (file) {
            fs.stat(path.join(dir, file), function (err, s) {
              if (err) {
                errs.push(err);
              }
              else if (s.isDirectory()) {
                dirs.push([file, s]);
              }
              else {
                files.push([file, s]);
              }

              if (--pending === 0) {
                cb(errs, dirs, files);
              }
            });
          });
        }

        function render(dirs, files) {
          // each entry in the array is a [name, stat] tuple

          // TODO: use stylessheets?
          var html = '<!doctype html>\
            <html> \
              <head> \
                <meta charset="utf-8"> \
                <title>Index of ' + pathname +'</title> \
              </head> \
              <body> \
            <h1>Index of ' + pathname + '</h1>\n';

          html += '<table>';

          var failed = false;
          var writeRow = function (file, i) {
            // render a row given a [name, stat] tuple
            var isDir = file[1].isDirectory();
            var href =
              parsed.pathname.replace(/\/$/, '') +
              '/' + encodeURIComponent(file[0]);

            // append trailing slash and query for dir entry
            if (isDir) {
              href += '/' + ((parsed.search)? parsed.search:'');
            }

            var displayName = ent.encode(file[0]) + ((isDir)? '/':'');

            // TODO: use stylessheets?
            html += '<tr>' +
              '<td><code>(' + permsToString(file[1]) + ')</code></td>' +
              '<td style="text-align: right; padding-left: 1em"><code>' + sizeToString(file[1], humanReadable, si) + '</code></td>' +
              '<td style="padding-left: 1em"><a href="' + href + '">' + displayName + '</a></td>' +
              '</tr>\n';
          };

          dirs.sort(function (a, b) { return b[0] - a[0]; } ).forEach(writeRow);
          files.sort(function (a, b) { return b.toString().localeCompare(a.toString()); }).forEach(writeRow);

          html += '</table>\n';
          html += '<br><address>Node.js ' +
            process.version +
            '/ <a href="https://github.com/jesusabdullah/node-ecstatic">ecstatic</a> ' +
            'server running @ ' +
            ent.encode(req.headers.host || '') + '</address>\n' +
            '</body></html>'
          ;

          if (!failed) {
            res.writeHead(200, { "Content-Type": "text/html" });
            res.end(html);
          }
        }
      });
    });
  };
};

function permsToString(stat) {
  var dir = stat.isDirectory() ? 'd' : '-',
      mode = stat.mode.toString(8);

  return dir + mode.slice(-3).split('').map(function (n) {
    return [
      '---',
      '--x',
      '-w-',
      '-wx',
      'r--',
      'r-x',
      'rw-',
      'rwx'
    ][parseInt(n, 10)];
  }).join('');
}

// given a file's stat, return the size of it in string
// humanReadable: (boolean) whether to result is human readable
// si: (boolean) whether to use si (1k = 1000), otherwise 1k = 1024
// adopted from http://stackoverflow.com/a/14919494/665507
function sizeToString(stat, humanReadable, si) {
    if (stat.isDirectory()) {
      return '';
    }

    var sizeString = '';
    var bytes = stat.size;
    var threshold = si ? 1000 : 1024;

    if(!humanReadable || bytes < threshold) {
      return bytes + 'B';
    }

    var units = ['k','M','G','T','P','E','Z','Y'];
    var u = -1;
    do {
        bytes /= threshold;
        ++u;
    } while (bytes >= threshold);
    return bytes.toFixed(1)+units[u];
}
