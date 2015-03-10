
NAME = 'PyYAML'
VERSION = '3.11'
DESCRIPTION = "YAML parser and emitter for Python"
LONG_DESCRIPTION = """\
YAML is a data serialization format designed for human readability
and interaction with scripting languages.  PyYAML is a YAML parser
and emitter for Python.

PyYAML features a complete YAML 1.1 parser, Unicode support, pickle
support, capable extension API, and sensible error messages.  PyYAML
supports standard YAML tags and provides Python-specific tags that
allow to represent an arbitrary Python object.

PyYAML is applicable for a broad range of tasks from complex
configuration files to object serialization and persistance."""
AUTHOR = "Kirill Simonov"
AUTHOR_EMAIL = 'xi@resolvent.net'
LICENSE = "MIT"
PLATFORMS = "Any"
URL = "http://pyyaml.org/wiki/PyYAML"
DOWNLOAD_URL = "http://pyyaml.org/download/pyyaml/%s-%s.tar.gz" % (NAME, VERSION)
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.5",
    "Programming Language :: Python :: 2.6",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.0",
    "Programming Language :: Python :: 3.1",
    "Programming Language :: Python :: 3.2",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Text Processing :: Markup",
]


LIBYAML_CHECK = """
#include <yaml.h>

int main(void) {
    yaml_parser_t parser;
    yaml_emitter_t emitter;

    yaml_parser_initialize(&parser);
    yaml_parser_delete(&parser);

    yaml_emitter_initialize(&emitter);
    yaml_emitter_delete(&emitter);

    return 0;
}
"""


import sys, os.path

from distutils import log
from distutils.core import setup, Command
from distutils.core import Distribution as _Distribution
from distutils.core import Extension as _Extension
from distutils.dir_util import mkpath
from distutils.command.build_ext import build_ext as _build_ext
from distutils.command.bdist_rpm import bdist_rpm as _bdist_rpm
from distutils.errors import CompileError, LinkError, DistutilsPlatformError

if 'setuptools.extension' in sys.modules:
    _Extension = sys.modules['setuptools.extension']._Extension
    sys.modules['distutils.core'].Extension = _Extension
    sys.modules['distutils.extension'].Extension = _Extension
    sys.modules['distutils.command.build_ext'].Extension = _Extension

with_pyrex = None
if sys.version_info[0] < 3:
    try:
        from Cython.Distutils.extension import Extension as _Extension
        from Cython.Distutils import build_ext as _build_ext
        with_pyrex = 'cython'
    except ImportError:
        try:
            # Pyrex cannot build _yaml.c at the moment,
            # but it may get fixed eventually.
            from Pyrex.Distutils import Extension as _Extension
            from Pyrex.Distutils import build_ext as _build_ext
            with_pyrex = 'pyrex'
        except ImportError:
            pass


class Distribution(_Distribution):

    def __init__(self, attrs=None):
        _Distribution.__init__(self, attrs)
        if not self.ext_modules:
            return
        for idx in range(len(self.ext_modules)-1, -1, -1):
            ext = self.ext_modules[idx]
            if not isinstance(ext, Extension):
                continue
            setattr(self, ext.attr_name, None)
            self.global_options = [
                    (ext.option_name, None,
                        "include %s (default if %s is available)"
                        % (ext.feature_description, ext.feature_name)),
                    (ext.neg_option_name, None,
                        "exclude %s" % ext.feature_description),
            ] + self.global_options
            self.negative_opt = self.negative_opt.copy()
            self.negative_opt[ext.neg_option_name] = ext.option_name

    def has_ext_modules(self):
        if not self.ext_modules:
            return False
        for ext in self.ext_modules:
            with_ext = self.ext_status(ext)
            if with_ext is None or with_ext:
                return True
        return False

    def ext_status(self, ext):
        if 'Java' in sys.version or 'IronPython' in sys.version or 'PyPy' in sys.version:
            return False
        if isinstance(ext, Extension):
            with_ext = getattr(self, ext.attr_name)
            return with_ext
        else:
            return True


class Extension(_Extension):

    def __init__(self, name, sources, feature_name, feature_description,
            feature_check, **kwds):
        if not with_pyrex:
            for filename in sources[:]:
                base, ext = os.path.splitext(filename)
                if ext == '.pyx':
                    sources.remove(filename)
                    sources.append('%s.c' % base)
        _Extension.__init__(self, name, sources, **kwds)
        self.feature_name = feature_name
        self.feature_description = feature_description
        self.feature_check = feature_check
        self.attr_name = 'with_' + feature_name.replace('-', '_')
        self.option_name = 'with-' + feature_name
        self.neg_option_name = 'without-' + feature_name


class build_ext(_build_ext):

    def run(self):
        optional = True
        disabled = True
        for ext in self.extensions:
            with_ext = self.distribution.ext_status(ext)
            if with_ext is None:
                disabled = False
            elif with_ext:
                optional = False
                disabled = False
                break
        if disabled:
            return
        try:
            _build_ext.run(self)
        except DistutilsPlatformError:
            exc = sys.exc_info()[1]
            if optional:
                log.warn(str(exc))
                log.warn("skipping build_ext")
            else:
                raise

    def get_source_files(self):
        self.check_extensions_list(self.extensions)
        filenames = []
        for ext in self.extensions:
            if with_pyrex == 'pyrex':
                self.pyrex_sources(ext.sources, ext)
            elif with_pyrex == 'cython':
                self.cython_sources(ext.sources, ext)
            for filename in ext.sources:
                filenames.append(filename)
                base = os.path.splitext(filename)[0]
                for ext in ['c', 'h', 'pyx', 'pxd']:
                    filename = '%s.%s' % (base, ext)
                    if filename not in filenames and os.path.isfile(filename):
                        filenames.append(filename)
        return filenames

    def get_outputs(self):
        self.check_extensions_list(self.extensions)
        outputs = []
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = os.path.join(self.build_lib,
                                    self.get_ext_filename(fullname))
            if os.path.isfile(filename):
                outputs.append(filename)
        return outputs

    def build_extensions(self):
        self.check_extensions_list(self.extensions)
        for ext in self.extensions:
            with_ext = self.distribution.ext_status(ext)
            if with_ext is None:
                with_ext = self.check_extension_availability(ext)
            if not with_ext:
                continue
            if with_pyrex == 'pyrex':
                ext.sources = self.pyrex_sources(ext.sources, ext)
            elif with_pyrex == 'cython':
                ext.sources = self.cython_sources(ext.sources, ext)
            self.build_extension(ext)

    def check_extension_availability(self, ext):
        cache = os.path.join(self.build_temp, 'check_%s.out' % ext.feature_name)
        if not self.force and os.path.isfile(cache):
            data = open(cache).read().strip()
            if data == '1':
                return True
            elif data == '0':
                return False
        mkpath(self.build_temp)
        src = os.path.join(self.build_temp, 'check_%s.c' % ext.feature_name)
        open(src, 'w').write(ext.feature_check)
        log.info("checking if %s is compilable" % ext.feature_name)
        try:
            [obj] = self.compiler.compile([src],
                    macros=ext.define_macros+[(undef,) for undef in ext.undef_macros],
                    include_dirs=ext.include_dirs,
                    extra_postargs=(ext.extra_compile_args or []),
                    depends=ext.depends)
        except CompileError:
            log.warn("")
            log.warn("%s is not found or a compiler error: forcing --%s"
                     % (ext.feature_name, ext.neg_option_name))
            log.warn("(if %s is installed correctly, you may need to"
                    % ext.feature_name)
            log.warn(" specify the option --include-dirs or uncomment and")
            log.warn(" modify the parameter include_dirs in setup.cfg)")
            open(cache, 'w').write('0\n')
            return False
        prog = 'check_%s' % ext.feature_name
        log.info("checking if %s is linkable" % ext.feature_name)
        try:
            self.compiler.link_executable([obj], prog,
                    output_dir=self.build_temp,
                    libraries=ext.libraries,
                    library_dirs=ext.library_dirs,
                    runtime_library_dirs=ext.runtime_library_dirs,
                    extra_postargs=(ext.extra_link_args or []))
        except LinkError:
            log.warn("")
            log.warn("%s is not found or a linker error: forcing --%s"
                     % (ext.feature_name, ext.neg_option_name))
            log.warn("(if %s is installed correctly, you may need to"
                    % ext.feature_name)
            log.warn(" specify the option --library-dirs or uncomment and")
            log.warn(" modify the parameter library_dirs in setup.cfg)")
            open(cache, 'w').write('0\n')
            return False
        open(cache, 'w').write('1\n')
        return True


class bdist_rpm(_bdist_rpm):

    def _make_spec_file(self):
        argv0 = sys.argv[0]
        features = []
        for ext in self.distribution.ext_modules:
            if not isinstance(ext, Extension):
                continue
            with_ext = getattr(self.distribution, ext.attr_name)
            if with_ext is None:
                continue
            if with_ext:
                features.append('--'+ext.option_name)
            else:
                features.append('--'+ext.neg_option_name)
        sys.argv[0] = ' '.join([argv0]+features)
        spec_file = _bdist_rpm._make_spec_file(self)
        sys.argv[0] = argv0
        return spec_file


class test(Command):

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        build_cmd = self.get_finalized_command('build')
        build_cmd.run()
        sys.path.insert(0, build_cmd.build_lib)
        if sys.version_info[0] < 3:
            sys.path.insert(0, 'tests/lib')
        else:
            sys.path.insert(0, 'tests/lib3')
        import test_all
        test_all.main([])


if __name__ == '__main__':

    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        license=LICENSE,
        platforms=PLATFORMS,
        url=URL,
        download_url=DOWNLOAD_URL,
        classifiers=CLASSIFIERS,

        package_dir={'': {2: 'lib', 3: 'lib3'}[sys.version_info[0]]},
        packages=['yaml'],
        ext_modules=[
            Extension('_yaml', ['ext/_yaml.pyx'],
                'libyaml', "LibYAML bindings", LIBYAML_CHECK,
                libraries=['yaml']),
        ],

        distclass=Distribution,

        cmdclass={
            'build_ext': build_ext,
            'bdist_rpm': bdist_rpm,
            'test': test,
        },
    )

