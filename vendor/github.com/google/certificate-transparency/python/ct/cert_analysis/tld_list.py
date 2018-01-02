import gflags
import logging
import re

FLAGS = gflags.FLAGS

gflags.DEFINE_string("tld_list_dir", "/tmp", "Stores top level domains list,"
                     " so it doesn't have to be fetched every run")

TLD_LIST_ADDR = "https://publicsuffix.org/list/effective_tld_names.dat"


class TLDList(object):
    """Contains list of top-level domains"""
    NOT_ADDRESS_REGEX = re.compile('[^a-z0-9\-.*]')
    def __init__(self, tld_dir=None, tld_file_name='tld_list'):
        if tld_dir is None:
            tld_dir = FLAGS.tld_list_dir

        tld_file = '/'.join((tld_dir, tld_file_name))
        try:
            with open(tld_file, 'r') as f:
                raw_list = f.read()
        except IOError:
            logging.warning("Couldn't open file with top level domains, "
                            "all matches will fail.")
            self.tld_tree = {}
        else:
            lines = unicode(raw_list, 'utf-8').splitlines()
            lines = lines[0:lines.index('// ===END ICANN DOMAINS===')]
            lines = filter(
                lambda line: not line.startswith('//') and not len(line) == 0,
                lines)
            lines = [line.split('.') for line in lines]
            self.tld_tree = {}
            for tld in lines:
                sub_tree = self.tld_tree
                for part in reversed(tld):
                    if part.startswith('*'):
                       sub_tree['*'] = []
                    elif part.startswith('!'):
                        sub_tree['*'].append(part[1:])
                    elif part not in sub_tree:
                        sub_tree[part] = {}
                        sub_tree = sub_tree[part]
                    else:
                        sub_tree = sub_tree[part]

    def match(self, address):
        """Matches address to the list.
        Returns:
            matching tld or None."""
        parts = address.split('.')
        best = []
        sub_tree = self.tld_tree
        for part in reversed(parts):
            if part in sub_tree:
                best.append(part)
                sub_tree = sub_tree[part]
            elif part not in sub_tree and '*' not in sub_tree:
                break
            elif '*' in sub_tree:
                for exception in sub_tree['*']:
                    if part == exception:
                        break
                    else:
                        best.append(part)
                # wildcard means that we can't go deeper
                break
        if best:
            best = '.'.join(reversed(best))
        else:
            best = None
        return best

    def match_idna(self, address):
        """Decodes address from idna and then matches to the list.
        Returns:
            matching tld or None."""
        try:
            idna = address.decode('idna')
            return self.match(idna)
        except:
            return None

    def match_certificate_name(self, name):
        """Returns both match and match_idna result for given address from cert.
        Also if name fails to be encoded as utf-8 third value in tuple will
        indicate that.
        Returns:
            tuple containing result from match and match_idna and boolean which
            indicates whether name successfully encoded in unicode.

        Raises:
            ValueError if name is not an address"""
        uni_fail = False
        try:
            uni_name = unicode(name, 'utf-8')
        except UnicodeError:
            # There could be some non-utf encoding with some characters
            # still matching a TLD so we move on, and try to match it.
            # RFC5280 says that dNSNames should be IA5String, but somebody still
            # could put utf-8 inside so it's reasonable to go through anyway.
            uni_fail = True
        else:
            # If an address fails this matching then it shouldn't work in any
            # browser, and trying to match it to TLDs will mess the output.
            uni_name = uni_name.lower()
            idna_name = uni_name.encode('idna')
            if self.NOT_ADDRESS_REGEX.search(idna_name):
                raise ValueError("%s (idna: %s) is not an address" % (uni_name,
                                                                      idna_name))
            name = uni_name
        name = name.lower()
        tld_match = self.match(name)
        # url could be decoded as idna
        idna_match = self.match_idna(name)
        return (tld_match, idna_match, uni_fail)

