// Fix: StorageVersionMigration should preserve storedVersions on failure
func (m *StorageVersionMigration) applyMigration() error {
    // Save original storedVersions for rollback
    originalVersions := m.crd.Status.StoredVersions
    
    // Attempt migration
    if err := m.migrate(); err != nil {
        // Fix: restore storedVersions on failed migration
        m.crd.Status.StoredVersions = originalVersions
        return err
    }
    return nil
}
