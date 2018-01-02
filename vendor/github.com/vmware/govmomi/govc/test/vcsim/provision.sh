PATH=$PATH:/sbin:/usr/sbin:/usr/local/sbin

echo "Creating vcsim model..."
cat > /etc/vmware-vpx/vcsim/model/initInventory-govc.cfg <<EOF
<config>
  <inventory>
    <dc>2</dc>
    <host-per-dc>3</host-per-dc>
    <vm-per-host>3</vm-per-host>
    <poweron-vm-per-host>2</poweron-vm-per-host>
    <cluster-per-dc>2</cluster-per-dc>
    <host-per-cluster>3</host-per-cluster>
    <rp-per-cluster>3</rp-per-cluster>
    <vm-per-rp>4</vm-per-rp>
    <poweron-vm-per-rp>3</poweron-vm-per-rp>
    <dv-portgroups>2</dv-portgroups>
  </inventory>
</config>
EOF

cat > /etc/vmware-vpx/vcsim/model/vcsim-default.cfg <<EOF
<simulator>
<enabled>true</enabled>
<initInventory>vcsim/model/initInventory-govc.cfg</initInventory>
</simulator>
EOF

echo "Starting VC simulator..."
vmware-vcsim-stop
vmware-vcsim-start default
