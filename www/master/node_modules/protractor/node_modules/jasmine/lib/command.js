var path = require('path'),
    fs = require('fs');

exports = module.exports = Command;

var subCommands = {
  init: {
    description: 'initialize jasmine',
    action: initJasmine
  },
  examples: {
    description: 'install examples',
    action: installExamples
  },
  help: {
    description: 'show help',
    action: help
  }
};

function Command(projectBaseDir) {
  this.projectBaseDir = projectBaseDir;
  this.specDir = path.join(projectBaseDir, 'spec');

  var command = this;

  this.run = function(jasmine, commands) {
    setEnvironmentVariables(commands);

    var commandToRun;
    Object.keys(subCommands).forEach(function(cmd) {
      if (commands.indexOf(cmd) >= 0) {
        commandToRun = subCommands[cmd];
      }
    });

    if (commandToRun) {
      commandToRun.action(command.projectBaseDir, command.specDir);
    } else {
      runJasmine(jasmine, parseOptions(commands));
    }
  };
}

function isFileArg(arg) {
  return arg.indexOf('--') !== 0 && !isEnvironmentVariable(arg);
}

function parseOptions(argv) {
  var files = [],
      color = true;
  argv.forEach(function(arg) {
    if (arg === '--no-color') {
      color = false;
    } else if (isFileArg(arg)) {
      files.push(arg);
    }
  });
  return {
    color: color,
    files: files
  };
}

function runJasmine(jasmine, env) {
  jasmine.loadConfigFile(process.env.JASMINE_CONFIG_PATH);

  jasmine.configureDefaultReporter({
    showColors: env.color
  });
  jasmine.execute(env.files);
}

function initJasmine(projectBaseDir, spec) {
  makeDirStructure(path.join(spec, 'support/'));
  if(!fs.existsSync(path.join(spec, 'support/jasmine.json'))) {
    fs.writeFileSync(path.join(spec, 'support/jasmine.json'), fs.readFileSync(path.join(__dirname, '../lib/examples/jasmine.json'), 'utf-8'));
  }
  else {
    console.log('spec/support/jasmine.json already exists in your project.');
  }
}

function installExamples(projectBaseDir, spec) {
  var jasmine_core_examples = path.join(__dirname, '../', 'node_modules/', 'jasmine-core/', 'lib/',
    'jasmine-core/', 'example/', 'node_example/');

  makeDirStructure(path.join(spec, 'support/'));
  makeDirStructure(path.join(spec, 'jasmine_examples/'));
  makeDirStructure(path.join(projectBaseDir, 'jasmine_examples/'));
  makeDirStructure(path.join(spec, 'helpers/jasmine_examples/'));

  copyFiles(path.join(jasmine_core_examples, 'spec/'), path.join(spec, 'helpers/', 'jasmine_examples/'), new RegExp(/[Hh]elper\.js/));
  copyFiles(path.join(jasmine_core_examples, 'src/'), path.join(projectBaseDir, 'jasmine_examples/'), new RegExp(/\.js/));
  copyFiles(path.join(jasmine_core_examples, 'spec/'), path.join(spec, 'jasmine_examples/'), new RegExp(/[Ss]pec.js/));
}

function help() {
  console.log('Usage: jasmine [command] [options] [files]');
  console.log('');
  console.log('Commands:');
  Object.keys(subCommands).forEach(function(cmd) {
    console.log('%s\t%s', lPad(cmd, 10), subCommands[cmd].description);
  });
  console.log('');

  console.log('Options:');
  console.log('%s\t\tturn off color in spec output', lPad('--no-color', 15));
  console.log('');
  console.log('if no command is given, jasmine specs will be run');
}

function lPad(str, length) {
  if (str.length === length) {
    return str;
  } else {
    return lPad(' ' + str, length);
  }
}

function copyFiles(srcDir, destDir, pattern) {
  var srcDirFiles = fs.readdirSync(srcDir);
  srcDirFiles.forEach(function(file) {
    if (file.search(pattern) !== -1) {
      fs.writeFileSync(path.join(destDir, file), fs.readFileSync(path.join(srcDir, file)));
    }
  });
}

function makeDirStructure(absolutePath) {
  var splitPath = absolutePath.split(path.sep);
  splitPath.forEach(function(dir, index) {
    if(index > 1) {
      var fullPath = path.join(splitPath.slice(0, index).join('/'), dir);
      if (!fs.existsSync(fullPath)) {
        fs.mkdirSync(fullPath);
      }
    }
  });
}

function isEnvironmentVariable(command) {
  var envRegExp = /(.*)=(.*)/;
  return command.match(envRegExp);
}

function setEnvironmentVariables(commands) {
  commands.forEach(function (command) {
    var regExpMatch = isEnvironmentVariable(command);
    if(regExpMatch) {
      var key = regExpMatch[1];
      var value = regExpMatch[2];
      process.env[key] = value;
    }
  });
}
