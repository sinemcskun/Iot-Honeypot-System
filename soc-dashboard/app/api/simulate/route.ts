export const dynamic = "force-dynamic";

const HOSTNAME = "svr04";
const KERNEL_VERSION = "3.2.0-4-amd64";

function uname(args: string[]): string {
  const all = args.some(a => a.includes("a"));
  if (all || args.length === 0)
    return all
      ? `Linux ${HOSTNAME} ${KERNEL_VERSION} #1 SMP Debian 3.2.68-1+deb7u1 x86_64 GNU/Linux`
      : "Linux";
  const parts: string[] = [];
  if (args.some(a => a.includes("s"))) parts.push("Linux");
  if (args.some(a => a.includes("n"))) parts.push(HOSTNAME);
  if (args.some(a => a.includes("r"))) parts.push(KERNEL_VERSION);
  if (args.some(a => a.includes("m"))) parts.push("x86_64");
  return parts.join(" ") || "Linux";
}

function ping(args: string[]): string {
  const host = args.find(a => !a.startsWith("-")) ?? "localhost";
  return `PING ${host} (93.184.216.34) 56(84) bytes of data.\n64 bytes from ${host}: icmp_seq=1 ttl=50 time=42.3 ms\n64 bytes from ${host}: icmp_seq=2 ttl=50 time=43.1 ms\n\n--- ${host} ping statistics ---\n2 packets transmitted, 2 received, 0% packet loss\nrtt min/avg/max = 42.3/42.7/43.1 ms`;
}

function wget(args: string[]): string {
  const url = args.find(a => a.startsWith("http") || (!a.startsWith("-") && a.includes("."))) ?? "";
  if (!url) return "wget: missing URL\nUsage: wget [OPTION]... [URL]...";
  const file = url.split("/").pop() || "index.html";
  return `--2026-03-12 14:32:07--  ${url}\nResolving ${url.split("/")[2] ?? url}... 93.184.216.34\nConnecting... connected.\nHTTP request sent... 200 OK\nLength: 45321 (44K)\nSaving to: '${file}'\n\n${file}    100%[=====>]  44.26K  128KB/s\n\n2026-03-12 14:32:08 (128 KB/s) - '${file}' saved [45321/45321]`;
}

function curl(args: string[]): string {
  const url = args.find(a => a.startsWith("http") || (!a.startsWith("-") && a.includes("."))) ?? "";
  if (!url) return "curl: try 'curl --help' for more information";
  return `  % Total    % Received % Xferd  Average Speed\n100 45321  100 45321    0     0   128k      0 --:--:-- --:--:-- --:--:--  128k`;
}

const STATIC: Record<string, string> = {
  whoami:   "root",
  id:       "uid=0(root) gid=0(root) groups=0(root)",
  hostname: HOSTNAME,
  pwd:      "/root",
  w:        ` 14:32:07 up 44 days, 13:40,  1 user,  load average: 0.08, 0.02, 0.01\nUSER     TTY      FROM              LOGIN@   IDLE WHAT\nroot     pts/0    192.168.1.100     14:31    0.00s w`,
  uptime:   " 14:32:07 up 44 days, 13:40,  1 user,  load average: 0.08, 0.02, 0.01",
  ifconfig: "eth0  Link encap:Ethernet  HWaddr aa:bb:cc:dd:ee:ff\n      inet addr:192.168.1.10  Mask:255.255.255.0\n      UP BROADCAST RUNNING  MTU:1500\n\nlo    inet addr:127.0.0.1  Mask:255.0.0.0",
  netstat:  "Active Internet connections\nProto Local Address      Foreign Address    State\ntcp   0.0.0.0:22         0.0.0.0:*          LISTEN\ntcp   192.168.1.10:22    192.168.1.100:512  ESTABLISHED",
  ps:       "  PID TTY          TIME CMD\n  420 pts/0    00:00:00 bash\n  523 pts/0    00:00:00 ps",
  nproc:    "2",
  groups:   "root",
  passwd:   "passwd: password updated successfully",
  cd:       "",
  mkdir:    "",
  touch:    "",
  rm:       "",
  cp:       "",
  mv:       "",
  chmod:    "",
  chown:    "",
  export:   "",
  grep:     "",
  awk:      "",
  sed:      "",
  kill:     "",
  pkill:    "",
  nohup:    "",
  sleep:    "",
  tar:      "",
  dd:       "",
  nc:       "",
  "apt-get":"Reading package lists... Done",
  apt:      "Reading package lists... Done",
};

function getResponse(command: string): string {
  const trimmed = command.trim();
  if (!trimmed) return "";

  let base = trimmed.split(/\s+/)[0];
  base = base.split("/").pop() ?? base;
  const args = trimmed.split(/\s+/).slice(1);

  if (base === "uname")    return uname(args);
  if (base === "ping")     return ping(args);
  if (base === "wget")     return wget(args);
  if (base === "curl")     return curl(args);
  if (base === "echo")     return args.join(" ").replace(/^["']|["']$/g, "");
  if (base === "ls") {
    const long = args.some(a => a.includes("l") || a.includes("a"));
    return long
      ? "total 28\ndrwxr-xr-x  2 root root 4096 Mar 10 ..\n-rw-------  1 root root  570 Mar 10 .bash_history\n-rw-r--r--  1 root root 3106 Mar 10 .bashrc"
      : ".bash_history  .bashrc  .profile";
  }
  if (base === "free") {
    const mb = args.some(a => a.includes("m"));
    return mb
      ? "              total   used   free\nMem:           3959   1214    974\nSwap:          2048     68   1979"
      : "              total       used       free\nMem:        4054744    1214880     997740\nSwap:       2097148      70632    2026516";
  }
  if (base === "crontab") return args.some(a => a.includes("l")) ? "no crontab for root" : "";
  if (base === "which")   return args[0] ? `/usr/bin/${args[0]}` : "";
  if (base === "sudo")    return args.length ? getResponse(args.join(" ")) : "";

  if (base in STATIC) return STATIC[base];

  return `-bash: ${base}: command not found`;
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const command = url.searchParams.get("cmd") ?? "";
  if (!command.trim()) {
    return Response.json({ error: "missing cmd parameter" }, { status: 400 });
  }

  const parts = command.split(/&&|;/).map(s => s.trim()).filter(Boolean);
  const outputs = parts.map(p => getResponse(p)).filter(o => o !== "");
  const response = outputs.join("\n") || "";

  return Response.json({ command, response });
}
