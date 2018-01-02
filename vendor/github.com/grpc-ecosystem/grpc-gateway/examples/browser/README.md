# Browser example

This directory contains an example use of grpc-gateway with web browsers.
The following commands automatically runs integration tests with phantomjs.

```shell-session
$ npm install -g gulp-cli
$ npm install
$ gulp
```

## Other examples

### Very simple example
Run
```shell-session
$ gulp bower
$ gulp backends
```

then, open `index.html`.


### Integration test with your browser

Run
```shell-session
$ gulp serve
```

then, open `http://localhost:8000` with your browser.
