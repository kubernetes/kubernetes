"use strict";

function Level(level, levelStr) {
  this.level = level;
  this.levelStr = levelStr;
}

/**
 * converts given String to corresponding Level
 * @param {String} sArg String value of Level OR Log4js.Level
 * @param {Log4js.Level} defaultLevel default Level, if no String representation
 * @return Level object
 * @type Log4js.Level
 */
function toLevel(sArg, defaultLevel) {

  if (!sArg) {
    return defaultLevel;
  }

  if (typeof sArg == "string") {
    var s = sArg.toUpperCase();
    if (module.exports[s]) {
      return module.exports[s];
    } else {
      return defaultLevel;
    }
  }

  return toLevel(sArg.toString());
}

Level.prototype.toString = function() {
  return this.levelStr;
};

Level.prototype.isLessThanOrEqualTo = function(otherLevel) {
  if (typeof otherLevel === "string") {
    otherLevel = toLevel(otherLevel);
  }
  return this.level <= otherLevel.level;
};

Level.prototype.isGreaterThanOrEqualTo = function(otherLevel) {
  if (typeof otherLevel === "string") {
    otherLevel = toLevel(otherLevel);
  }
  return this.level >= otherLevel.level;
};

Level.prototype.isEqualTo = function(otherLevel) {
  if (typeof otherLevel == "string") {
    otherLevel = toLevel(otherLevel);
  }
  return this.level === otherLevel.level;
};

module.exports = {
  ALL: new Level(Number.MIN_VALUE, "ALL"), 
  TRACE: new Level(5000, "TRACE"), 
  DEBUG: new Level(10000, "DEBUG"), 
  INFO: new Level(20000, "INFO"), 
  WARN: new Level(30000, "WARN"), 
  ERROR: new Level(40000, "ERROR"), 
  FATAL: new Level(50000, "FATAL"), 
  MARK: new Level(9007199254740992, "MARK"), // 2^53
  OFF: new Level(Number.MAX_VALUE, "OFF"), 
  toLevel: toLevel
};
