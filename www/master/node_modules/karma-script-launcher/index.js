var ScriptBrowser = function(baseBrowserDecorator, script) {
  baseBrowserDecorator(this);

  this.name = script;

  this._getCommand = function() {
    return script;
  };
};

ScriptBrowser.$inject = ['baseBrowserDecorator', 'name'];


// PUBLISH DI MODULE
module.exports = {
  'launcher:Script': ['type', ScriptBrowser]
};
