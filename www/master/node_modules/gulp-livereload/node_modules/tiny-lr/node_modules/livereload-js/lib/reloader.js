(function() {
  var IMAGE_STYLES, Reloader, numberOfMatchingSegments, pathFromUrl, pathsMatch, pickBestMatch, splitUrl;

  splitUrl = function(url) {
    var hash, index, params;
    if ((index = url.indexOf('#')) >= 0) {
      hash = url.slice(index);
      url = url.slice(0, index);
    } else {
      hash = '';
    }
    if ((index = url.indexOf('?')) >= 0) {
      params = url.slice(index);
      url = url.slice(0, index);
    } else {
      params = '';
    }
    return {
      url: url,
      params: params,
      hash: hash
    };
  };

  pathFromUrl = function(url) {
    var path;
    url = splitUrl(url).url;
    if (url.indexOf('file://') === 0) {
      path = url.replace(/^file:\/\/(localhost)?/, '');
    } else {
      path = url.replace(/^([^:]+:)?\/\/([^:\/]+)(:\d*)?\//, '/');
    }
    return decodeURIComponent(path);
  };

  pickBestMatch = function(path, objects, pathFunc) {
    var bestMatch, object, score, _i, _len;
    bestMatch = {
      score: 0
    };
    for (_i = 0, _len = objects.length; _i < _len; _i++) {
      object = objects[_i];
      score = numberOfMatchingSegments(path, pathFunc(object));
      if (score > bestMatch.score) {
        bestMatch = {
          object: object,
          score: score
        };
      }
    }
    if (bestMatch.score > 0) {
      return bestMatch;
    } else {
      return null;
    }
  };

  numberOfMatchingSegments = function(path1, path2) {
    var comps1, comps2, eqCount, len;
    path1 = path1.replace(/^\/+/, '').toLowerCase();
    path2 = path2.replace(/^\/+/, '').toLowerCase();
    if (path1 === path2) {
      return 10000;
    }
    comps1 = path1.split('/').reverse();
    comps2 = path2.split('/').reverse();
    len = Math.min(comps1.length, comps2.length);
    eqCount = 0;
    while (eqCount < len && comps1[eqCount] === comps2[eqCount]) {
      ++eqCount;
    }
    return eqCount;
  };

  pathsMatch = function(path1, path2) {
    return numberOfMatchingSegments(path1, path2) > 0;
  };

  IMAGE_STYLES = [
    {
      selector: 'background',
      styleNames: ['backgroundImage']
    }, {
      selector: 'border',
      styleNames: ['borderImage', 'webkitBorderImage', 'MozBorderImage']
    }
  ];

  exports.Reloader = Reloader = (function() {
    function Reloader(window, console, Timer) {
      this.window = window;
      this.console = console;
      this.Timer = Timer;
      this.document = this.window.document;
      this.importCacheWaitPeriod = 200;
      this.plugins = [];
    }

    Reloader.prototype.addPlugin = function(plugin) {
      return this.plugins.push(plugin);
    };

    Reloader.prototype.analyze = function(callback) {
      return results;
    };

    Reloader.prototype.reload = function(path, options) {
      var plugin, _base, _i, _len, _ref;
      this.options = options;
      if ((_base = this.options).stylesheetReloadTimeout == null) {
        _base.stylesheetReloadTimeout = 15000;
      }
      _ref = this.plugins;
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        plugin = _ref[_i];
        if (plugin.reload && plugin.reload(path, options)) {
          return;
        }
      }
      if (options.liveCSS) {
        if (path.match(/\.css$/i)) {
          if (this.reloadStylesheet(path)) {
            return;
          }
        }
      }
      if (options.liveImg) {
        if (path.match(/\.(jpe?g|png|gif)$/i)) {
          this.reloadImages(path);
          return;
        }
      }
      return this.reloadPage();
    };

    Reloader.prototype.reloadPage = function() {
      return this.window.document.location.reload();
    };

    Reloader.prototype.reloadImages = function(path) {
      var expando, img, selector, styleNames, styleSheet, _i, _j, _k, _l, _len, _len1, _len2, _len3, _ref, _ref1, _ref2, _ref3, _results;
      expando = this.generateUniqueString();
      _ref = this.document.images;
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        img = _ref[_i];
        if (pathsMatch(path, pathFromUrl(img.src))) {
          img.src = this.generateCacheBustUrl(img.src, expando);
        }
      }
      if (this.document.querySelectorAll) {
        for (_j = 0, _len1 = IMAGE_STYLES.length; _j < _len1; _j++) {
          _ref1 = IMAGE_STYLES[_j], selector = _ref1.selector, styleNames = _ref1.styleNames;
          _ref2 = this.document.querySelectorAll("[style*=" + selector + "]");
          for (_k = 0, _len2 = _ref2.length; _k < _len2; _k++) {
            img = _ref2[_k];
            this.reloadStyleImages(img.style, styleNames, path, expando);
          }
        }
      }
      if (this.document.styleSheets) {
        _ref3 = this.document.styleSheets;
        _results = [];
        for (_l = 0, _len3 = _ref3.length; _l < _len3; _l++) {
          styleSheet = _ref3[_l];
          _results.push(this.reloadStylesheetImages(styleSheet, path, expando));
        }
        return _results;
      }
    };

    Reloader.prototype.reloadStylesheetImages = function(styleSheet, path, expando) {
      var e, rule, rules, styleNames, _i, _j, _len, _len1;
      try {
        rules = styleSheet != null ? styleSheet.cssRules : void 0;
      } catch (_error) {
        e = _error;
      }
      if (!rules) {
        return;
      }
      for (_i = 0, _len = rules.length; _i < _len; _i++) {
        rule = rules[_i];
        switch (rule.type) {
          case CSSRule.IMPORT_RULE:
            this.reloadStylesheetImages(rule.styleSheet, path, expando);
            break;
          case CSSRule.STYLE_RULE:
            for (_j = 0, _len1 = IMAGE_STYLES.length; _j < _len1; _j++) {
              styleNames = IMAGE_STYLES[_j].styleNames;
              this.reloadStyleImages(rule.style, styleNames, path, expando);
            }
            break;
          case CSSRule.MEDIA_RULE:
            this.reloadStylesheetImages(rule, path, expando);
        }
      }
    };

    Reloader.prototype.reloadStyleImages = function(style, styleNames, path, expando) {
      var newValue, styleName, value, _i, _len;
      for (_i = 0, _len = styleNames.length; _i < _len; _i++) {
        styleName = styleNames[_i];
        value = style[styleName];
        if (typeof value === 'string') {
          newValue = value.replace(/\burl\s*\(([^)]*)\)/, (function(_this) {
            return function(match, src) {
              if (pathsMatch(path, pathFromUrl(src))) {
                return "url(" + (_this.generateCacheBustUrl(src, expando)) + ")";
              } else {
                return match;
              }
            };
          })(this));
          if (newValue !== value) {
            style[styleName] = newValue;
          }
        }
      }
    };

    Reloader.prototype.reloadStylesheet = function(path) {
      var imported, link, links, match, style, _i, _j, _k, _l, _len, _len1, _len2, _len3, _ref, _ref1;
      links = (function() {
        var _i, _len, _ref, _results;
        _ref = this.document.getElementsByTagName('link');
        _results = [];
        for (_i = 0, _len = _ref.length; _i < _len; _i++) {
          link = _ref[_i];
          if (link.rel.match(/^stylesheet$/i) && !link.__LiveReload_pendingRemoval) {
            _results.push(link);
          }
        }
        return _results;
      }).call(this);
      imported = [];
      _ref = this.document.getElementsByTagName('style');
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        style = _ref[_i];
        if (style.sheet) {
          this.collectImportedStylesheets(style, style.sheet, imported);
        }
      }
      for (_j = 0, _len1 = links.length; _j < _len1; _j++) {
        link = links[_j];
        this.collectImportedStylesheets(link, link.sheet, imported);
      }
      if (this.window.StyleFix && this.document.querySelectorAll) {
        _ref1 = this.document.querySelectorAll('style[data-href]');
        for (_k = 0, _len2 = _ref1.length; _k < _len2; _k++) {
          style = _ref1[_k];
          links.push(style);
        }
      }
      this.console.log("LiveReload found " + links.length + " LINKed stylesheets, " + imported.length + " @imported stylesheets");
      match = pickBestMatch(path, links.concat(imported), (function(_this) {
        return function(l) {
          return pathFromUrl(_this.linkHref(l));
        };
      })(this));
      if (match) {
        if (match.object.rule) {
          this.console.log("LiveReload is reloading imported stylesheet: " + match.object.href);
          this.reattachImportedRule(match.object);
        } else {
          this.console.log("LiveReload is reloading stylesheet: " + (this.linkHref(match.object)));
          this.reattachStylesheetLink(match.object);
        }
      } else {
        this.console.log("LiveReload will reload all stylesheets because path '" + path + "' did not match any specific one");
        for (_l = 0, _len3 = links.length; _l < _len3; _l++) {
          link = links[_l];
          this.reattachStylesheetLink(link);
        }
      }
      return true;
    };

    Reloader.prototype.collectImportedStylesheets = function(link, styleSheet, result) {
      var e, index, rule, rules, _i, _len;
      try {
        rules = styleSheet != null ? styleSheet.cssRules : void 0;
      } catch (_error) {
        e = _error;
      }
      if (rules && rules.length) {
        for (index = _i = 0, _len = rules.length; _i < _len; index = ++_i) {
          rule = rules[index];
          switch (rule.type) {
            case CSSRule.CHARSET_RULE:
              continue;
            case CSSRule.IMPORT_RULE:
              result.push({
                link: link,
                rule: rule,
                index: index,
                href: rule.href
              });
              this.collectImportedStylesheets(link, rule.styleSheet, result);
              break;
            default:
              break;
          }
        }
      }
    };

    Reloader.prototype.waitUntilCssLoads = function(clone, func) {
      var callbackExecuted, executeCallback, poll;
      callbackExecuted = false;
      executeCallback = (function(_this) {
        return function() {
          if (callbackExecuted) {
            return;
          }
          callbackExecuted = true;
          return func();
        };
      })(this);
      clone.onload = (function(_this) {
        return function() {
          _this.console.log("LiveReload: the new stylesheet has finished loading");
          _this.knownToSupportCssOnLoad = true;
          return executeCallback();
        };
      })(this);
      if (!this.knownToSupportCssOnLoad) {
        (poll = (function(_this) {
          return function() {
            if (clone.sheet) {
              _this.console.log("LiveReload is polling until the new CSS finishes loading...");
              return executeCallback();
            } else {
              return _this.Timer.start(50, poll);
            }
          };
        })(this))();
      }
      return this.Timer.start(this.options.stylesheetReloadTimeout, executeCallback);
    };

    Reloader.prototype.linkHref = function(link) {
      return link.href || link.getAttribute('data-href');
    };

    Reloader.prototype.reattachStylesheetLink = function(link) {
      var clone, parent;
      if (link.__LiveReload_pendingRemoval) {
        return;
      }
      link.__LiveReload_pendingRemoval = true;
      if (link.tagName === 'STYLE') {
        clone = this.document.createElement('link');
        clone.rel = 'stylesheet';
        clone.media = link.media;
        clone.disabled = link.disabled;
      } else {
        clone = link.cloneNode(false);
      }
      clone.href = this.generateCacheBustUrl(this.linkHref(link));
      parent = link.parentNode;
      if (parent.lastChild === link) {
        parent.appendChild(clone);
      } else {
        parent.insertBefore(clone, link.nextSibling);
      }
      return this.waitUntilCssLoads(clone, (function(_this) {
        return function() {
          var additionalWaitingTime;
          if (/AppleWebKit/.test(navigator.userAgent)) {
            additionalWaitingTime = 5;
          } else {
            additionalWaitingTime = 200;
          }
          return _this.Timer.start(additionalWaitingTime, function() {
            var _ref;
            if (!link.parentNode) {
              return;
            }
            link.parentNode.removeChild(link);
            clone.onreadystatechange = null;
            return (_ref = _this.window.StyleFix) != null ? _ref.link(clone) : void 0;
          });
        };
      })(this));
    };

    Reloader.prototype.reattachImportedRule = function(_arg) {
      var href, index, link, media, newRule, parent, rule, tempLink;
      rule = _arg.rule, index = _arg.index, link = _arg.link;
      parent = rule.parentStyleSheet;
      href = this.generateCacheBustUrl(rule.href);
      media = rule.media.length ? [].join.call(rule.media, ', ') : '';
      newRule = "@import url(\"" + href + "\") " + media + ";";
      rule.__LiveReload_newHref = href;
      tempLink = this.document.createElement("link");
      tempLink.rel = 'stylesheet';
      tempLink.href = href;
      tempLink.__LiveReload_pendingRemoval = true;
      if (link.parentNode) {
        link.parentNode.insertBefore(tempLink, link);
      }
      return this.Timer.start(this.importCacheWaitPeriod, (function(_this) {
        return function() {
          if (tempLink.parentNode) {
            tempLink.parentNode.removeChild(tempLink);
          }
          if (rule.__LiveReload_newHref !== href) {
            return;
          }
          parent.insertRule(newRule, index);
          parent.deleteRule(index + 1);
          rule = parent.cssRules[index];
          rule.__LiveReload_newHref = href;
          return _this.Timer.start(_this.importCacheWaitPeriod, function() {
            if (rule.__LiveReload_newHref !== href) {
              return;
            }
            parent.insertRule(newRule, index);
            return parent.deleteRule(index + 1);
          });
        };
      })(this));
    };

    Reloader.prototype.generateUniqueString = function() {
      return 'livereload=' + Date.now();
    };

    Reloader.prototype.generateCacheBustUrl = function(url, expando) {
      var hash, oldParams, originalUrl, params, _ref;
      if (expando == null) {
        expando = this.generateUniqueString();
      }
      _ref = splitUrl(url), url = _ref.url, hash = _ref.hash, oldParams = _ref.params;
      if (this.options.overrideURL) {
        if (url.indexOf(this.options.serverURL) < 0) {
          originalUrl = url;
          url = this.options.serverURL + this.options.overrideURL + "?url=" + encodeURIComponent(url);
          this.console.log("LiveReload is overriding source URL " + originalUrl + " with " + url);
        }
      }
      params = oldParams.replace(/(\?|&)livereload=(\d+)/, function(match, sep) {
        return "" + sep + expando;
      });
      if (params === oldParams) {
        if (oldParams.length === 0) {
          params = "?" + expando;
        } else {
          params = "" + oldParams + "&" + expando;
        }
      }
      return url + params + hash;
    };

    return Reloader;

  })();

}).call(this);
