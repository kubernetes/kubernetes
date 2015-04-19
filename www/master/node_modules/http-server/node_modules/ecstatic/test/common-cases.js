module.exports = {
  'a.txt' : {
    code : 200,
    type : 'text/plain',
    body : 'A!!!\n',
  },
  'b.txt' : {
    code : 200,
    type : 'text/plain',
    body : 'B!!!\n',
  },
  'c.js' : {
    code : 200,
    type : 'application/javascript',
    body : 'console.log(\'C!!!\');\n',
  },
  'd.js' : {
    code : 200,
    type : 'application/javascript',
    body : 'console.log(\'C!!!\');\n',
  },
  'subdir/e.html' : {
    code : 200,
    type : 'text/html',
    body : '<b>e!!</b>\n',
  },
  // test for defaultExt
  'subdir/e?foo=bar' : {
    code : 200,
    type : 'text/html',
    body : '<b>e!!</b>\n',
  },
  // test for defaultExt with noisy query param
  'subdir/e?foo=bar.ext' : {
    code : 200,
    type : 'text/html',
    body : '<b>e!!</b>\n',
  },
  'subdir/index.html' : {
    code : 200,
    type : 'text/html',
    body : 'index!!!\n',
  },
  'subdir' : {
    code : 302,
    location: 'subdir/'
  },
  'subdir?foo=bar': {
    code: 302,
    location: 'subdir/?foo=bar'
  },
  // test for url-encoded paths
  '%E4%B8%AD%E6%96%87' : {  // '/中文'
    code : 302,
    location: '%E4%B8%AD%E6%96%87/'
  },
  '%E4%B8%AD%E6%96%87?%E5%A4%AB=%E5%B7%B4': {  // '中文?夫=巴'
    code: 302,
    location: '%E4%B8%AD%E6%96%87/?%E5%A4%AB=%E5%B7%B4'
  },
  'subdir/' : {
    code : 200,
    type : 'text/html',
    body : 'index!!!\n'
  },
  '404' : {
    code : 200,
    type : 'text/html',
    body : '<h1>404</h1>\n'
  },
  'something-non-existant' : {
    code : 200,
    type : 'text/html',
    body : '<h1>404</h1>\n'
  },
  'compress/foo.js' : {
    code : 200,
    file: 'compress/foo.js.gz',
    headers: {'accept-encoding': 'compress, gzip'}
  },
  // no accept-encoding of gzip, so serve regular file
  'compress/foo_2.js' : {
    code : 200,
    file: 'compress/foo_2.js'
  },
  'emptyDir/': {
    code: 200
  },
  'subdir_with space' : {
    code: 302,
    location: 'subdir_with%20space/'
  },
  'subdir_with space/index.html' : {
    code: 200,
    type: 'text/html',
    body: 'index :)\n'
  },
  'something-non-existant%00.png': {
    code: 200,
    type: 'text/html',
    body: '<h1>404</h1>\n'
  }
};

if (require.main === module) {
  console.log("ok 1 - test cases included");
}
