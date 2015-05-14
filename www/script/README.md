## Running integration tests

* `script/test-unit.sh` - run unit tests
* `script/test-integration.sh` - run integration specs

### Running tests headlessly

Start Xvfb and export DISPLAY variable:

```
./script/xvfb.sh start
export DISPLAY=:99
```
