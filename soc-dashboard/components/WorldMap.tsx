"use client";
import { useState } from "react";
import {
  ComposableMap,
  Geographies,
  Geography,
  Marker,
  ZoomableGroup,
} from "react-simple-maps";

const GEO_URL = "/world-110m.json";

interface GeoPoint {
  lat: number;
  lon: number;
  count: number;
  country: string;
  org: string;
}

interface WorldMapProps {
  points: GeoPoint[];
}

export default function WorldMap({ points }: WorldMapProps) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);
  const [zoom, setZoom] = useState(1);

  const maxCount = Math.max(...points.map(p => p.count), 1);

  function markerRadius(count: number): number {
    const min = 2;
    const max = 10;
    return min + (Math.sqrt(count) / Math.sqrt(maxCount)) * (max - min);
  }

  function markerOpacity(count: number): number {
    return 0.5 + (count / maxCount) * 0.5;
  }

  return (
    <div style={{ position: "relative", background: "var(--bg-card)", borderRadius: 8, width: "100%", height: "100%", overflow: "hidden" }}>
      <ComposableMap
        projection="geoMercator"
        projectionConfig={{ scale: 110, center: [0, 20] }}
        width={800}
        height={400}
        style={{ width: "100%", height: "100%", display: "block" }}
      >
        <ZoomableGroup zoom={zoom}>
          <Geographies geography={GEO_URL}>
            {({ geographies }) =>
              geographies.map(geo => (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  fill="#1e293b"
                  stroke="#334155"
                  strokeWidth={0.4}
                  style={{
                    default: { outline: "none" },
                    hover:   { outline: "none", fill: "#263548" },
                    pressed: { outline: "none" },
                  }}
                />
              ))
            }
          </Geographies>

          {points.map((point, i) => (
            <Marker
              key={i}
              coordinates={[point.lon, point.lat]}
              onMouseEnter={(e) => {
                const rect = (e.target as SVGElement)
                  .closest("svg")
                  ?.getBoundingClientRect();
                if (rect) {
                  setTooltip({
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top - 10,
                    content: `${point.country} — ${point.count.toLocaleString()} events\n${point.org}`,
                  });
                }
              }}
              onMouseLeave={() => setTooltip(null)}
            >
              <circle
                r={markerRadius(point.count) / Math.sqrt(zoom)}
                fill="#ef4444"
                fillOpacity={markerOpacity(point.count)}
                stroke="#fca5a5"
                strokeWidth={0.5}
                style={{ cursor: "pointer" }}
              />
            </Marker>
          ))}
        </ZoomableGroup>
      </ComposableMap>

      {/* Zoom buttons */}
      <div style={{
        position: "absolute", bottom: 10, right: 10,
        display: "flex", flexDirection: "column", gap: 4,
      }}>
        {[{ label: "+", action: () => setZoom(z => Math.min(z + 0.5, 6)) },
          { label: "−", action: () => setZoom(z => Math.max(z - 0.5, 1)) },
          { label: "⟳", action: () => setZoom(1) },
        ].map(btn => (
          <button
            key={btn.label}
            onClick={btn.action}
            style={{
              width: 28, height: 28,
              background: "#1e293b",
              border: "1px solid #334155",
              borderRadius: 6,
              color: "#94a3b8",
              fontSize: 14,
              cursor: "pointer",
              display: "flex", alignItems: "center", justifyContent: "center",
              transition: "background 0.15s, color 0.15s",
            }}
            onMouseEnter={e => {
              (e.currentTarget as HTMLButtonElement).style.background = "#334155";
              (e.currentTarget as HTMLButtonElement).style.color = "#f1f5f9";
            }}
            onMouseLeave={e => {
              (e.currentTarget as HTMLButtonElement).style.background = "#1e293b";
              (e.currentTarget as HTMLButtonElement).style.color = "#94a3b8";
            }}
          >
            {btn.label}
          </button>
        ))}
      </div>

      {tooltip && (
        <div
          style={{
            position: "absolute",
            left: tooltip.x + 12,
            top: tooltip.y,
            background: "#0f172a",
            border: "1px solid var(--border)",
            borderRadius: 6,
            padding: "8px 12px",
            fontSize: 12,
            color: "var(--text-primary)",
            pointerEvents: "none",
            whiteSpace: "pre-line",
            zIndex: 10,
            maxWidth: 220,
          }}
        >
          {tooltip.content}
        </div>
      )}
    </div>
  );
}
