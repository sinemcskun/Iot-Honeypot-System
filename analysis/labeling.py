import numpy as np
import pandas as pd


class AttackLabeler:
    LABEL_COLS = [
        "label_brute_force", "label_malware_dropper", "label_tunneling",
        "label_lateral_movement", "label_reconnaissance",
        "label_data_exfiltration", "label_destructive",
        "label_port_scan", "label_credential_spray",
        "label_service_interaction", "label_network_probe",
    ]


    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        print("[AttackLabeler] Labeling starting...")

        z = pd.Series(0, index=df.index)
        cmd_count = df.get("cmd_count", z.copy())
        cred = df.get("credential_attempts", z.copy())
        upwd = df.get("unique_passwords", z.copy())
        chain = df.get("has_download_execute_chain", z.copy())
        dl = df.get("has_downloader", z.copy())
        pers = df.get("has_persistence", z.copy())
        fwd = df.get("has_port_forwarding", z.copy())
        tls = df.get("tls_on_non_standard_port", z.copy())
        dns_t = df.get("dns_tunnel_indicator", z.copy())
        scan = df.get("has_internal_ip_scan", z.copy())
        recon = df.get("has_recon_commands", z.copy())
        exfil = df.get("has_file_exfiltration", z.copy())
        destr = df.get("has_destructive", z.copy())
        uports = df.get("unique_dest_ports", z.copy())

        df["label_brute_force"] = ((cred >= 3) & (upwd >= 2)).astype(int)
        df["label_malware_dropper"] = ((chain == 1) | ((dl == 1) & (pers == 1))).astype(int)
        df["label_tunneling"] = ((fwd == 1) | (tls == 1) | (dns_t == 1)).astype(int)
        df["label_lateral_movement"] = ((scan == 1) | ((fwd == 1) & (recon == 1))).astype(int)
        df["label_reconnaissance"] = (recon == 1).astype(int)
        df["label_data_exfiltration"] = (exfil == 1).astype(int)
        df["label_destructive"] = (destr == 1).astype(int)

        df["label_port_scan"] = (uports >= 3).astype(int)
        df["label_credential_spray"] = ((cred >= 1) & (df["label_brute_force"] == 0)).astype(int)

        specific = (
            df["label_brute_force"] + df["label_malware_dropper"] +
            df["label_tunneling"] + df["label_lateral_movement"] +
            df["label_reconnaissance"] + df["label_data_exfiltration"] +
            df["label_destructive"]
        )
        df["label_service_interaction"] = ((cmd_count > 0) & (specific == 0)).astype(int)

        all_prior = specific + df["label_port_scan"] + df["label_credential_spray"] + df["label_service_interaction"]
        df["label_network_probe"] = (all_prior == 0).astype(int)

        names = [c.replace("label_", "") for c in self.LABEL_COLS]
        df["attack_labels"] = df[self.LABEL_COLS].apply(
            lambda row: "|".join(n for n, v in zip(names, row) if v == 1), axis=1,
        )
        df["attack_label_count"] = df[self.LABEL_COLS].sum(axis=1).astype(int)

        for col in self.LABEL_COLS:
            print(f"  {col}: {df[col].sum():,}")
        unlabeled = (df["attack_label_count"] == 0).sum()
        print(f"[AttackLabeler] {df['attack_label_count'].gt(0).sum():,}/{len(df):,} labeled. Unlabeled: {unlabeled}")
        return df


class AutomationProfiler:
    def profile(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        print("[AutomationProfiler] Profiling starting...")

        var_ratio = df.get("timing_variance_ratio", pd.Series(0.5, index=df.index))
        epm = df.get("events_per_minute", pd.Series(0.0, index=df.index))
        cmd_ratio = df.get("cmd_unique_ratio", pd.Series(0.5, index=df.index))
        burst = df.get("burst_count", pd.Series(0, index=df.index))

        score = pd.Series(0.5, index=df.index)
        score = score - np.clip(var_ratio * 0.2, 0, 0.2)
        score = score + np.where(epm > 20, 0.2, np.where(epm > 5, 0.1, 0.0))
        score = score - np.clip(cmd_ratio * 0.2, 0, 0.2)
        score = score + np.where(burst > 3, 0.1, 0.0)

        df["automation_score"] = np.clip(score, 0.0, 1.0).round(2)
        df["automation_likelihood"] = np.where(
            df["automation_score"] >= 0.7, "scripted",
            np.where(df["automation_score"] <= 0.3, "manual", "mixed"),
        )

        counts = df["automation_likelihood"].value_counts()
        print(f"[AutomationProfiler] {dict(counts)}")
        return df
