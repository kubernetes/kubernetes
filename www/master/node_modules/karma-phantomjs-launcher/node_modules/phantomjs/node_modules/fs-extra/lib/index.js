var jsonFile = require('jsonfile')
var json = require('./json')

var fse = {}
var fs = require("graceful-fs")

//attach fs methods to fse
Object.keys(fs).forEach(function(key) {
  var func = fs[key]
  if (typeof func == 'function')
    fse[key] = func
})
fs = fse

var copy = require('./copy')
fs.copy = copy.copy
fs.copySync = copy.copySync

var remove = require('./remove')
fs.remove = remove.remove
fs.removeSync = remove.removeSync
fs['delete'] = fs.remove
fs.deleteSync = fs.removeSync

var mkdir = require('./mkdir')
fs.mkdirs = mkdir.mkdirs
fs.mkdirsSync = mkdir.mkdirsSync
fs.mkdirp = fs.mkdirs
fs.mkdirpSync = fs.mkdirsSync

var create = require('./create')
fs.createFile = create.createFile
fs.createFileSync = create.createFileSync

fs.ensureFile = create.createFile
fs.ensureFileSync = create.createFileSync
fs.ensureDir = mkdir.mkdirs
fs.ensureDirSync = mkdir.mkdirsSync


var move = require('./move')
fs.move = function(src, dest, opts, callback) {
  if (typeof opts == 'function') {
    callback = opts
    opts = {}
  }

  if (opts.mkdirp == null) opts.mkdirp = true
  if (opts.clobber == null) opts.clobber = false

  move(src, dest, opts, callback)
}


var output = require('./output')
fs.outputFile = output.outputFile
fs.outputFileSync = output.outputFileSync


fs.readJsonFile = jsonFile.readFile
fs.readJSONFile = jsonFile.readFile
fs.readJsonFileSync = jsonFile.readFileSync
fs.readJSONFileSync = jsonFile.readFileSync

fs.readJson = jsonFile.readFile
fs.readJSON = jsonFile.readFile
fs.readJsonSync = jsonFile.readFileSync
fs.readJSONSync = jsonFile.readFileSync

fs.outputJsonSync = json.outputJsonSync
fs.outputJSONSync = json.outputJsonSync
fs.outputJson = json.outputJson
fs.outputJSON = json.outputJson

fs.writeJsonFile = jsonFile.writeFile
fs.writeJSONFile = jsonFile.writeFile
fs.writeJsonFileSync = jsonFile.writeFileSync
fs.writeJSONFileSync = jsonFile.writeFileSync

fs.writeJson = jsonFile.writeFile
fs.writeJSON = jsonFile.writeFile
fs.writeJsonSync = jsonFile.writeFileSync
fs.writeJSONSync = jsonFile.writeFileSync


module.exports = fs

jsonFile.spaces = 2 //set to 2
module.exports.jsonfile = jsonFile //so users of fs-extra can modify jsonFile.spaces

