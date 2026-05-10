import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SOC Honeypot Dashboard",
  description: "Honeypot Behavioral Intelligence — IoT Threat Classification Platform",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full" style={{ background: "var(--bg-primary)" }}>
        {children}
      </body>
    </html>
  );
}
