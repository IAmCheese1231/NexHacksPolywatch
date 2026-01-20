"use client";
import React, { useEffect, useState } from "react";
import {
  Settings,
  GitBranch,
  PieChart,
  BarChart3,
} from "lucide-react";
import ScenarioTester from "./components/ScenarioTester";
import AlertsAnalytics from "./components/AlertsAnalytics";

const DEFAULT_PORTFOLIO_DASHBOARD_URL =
  process.env.NEXT_PUBLIC_PORTFOLIO_DASHBOARD_URL ||
  (process.env.NODE_ENV === "development" ? "http://localhost:5173" : "");

function Card({
  children,
  className = "",
}: {
  children?: React.ReactNode; // ✅ optional
  className?: string;
}) {
  return (
    <div
      className={`rounded-2xl border border-white/10 bg-white/[0.07] shadow-2xl shadow-black/20 ring-1 ring-white/5 backdrop-blur-xl ${className}`}
    >
      {children}
    </div>
  );
}

function CardHeader({
  title,
  right,
  icon,
}: {
  title: string;
  right?: React.ReactNode;
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-white/10 px-5 py-4">
      <div className="flex items-center gap-3">
        {icon ? (
          <div className="grid h-9 w-9 place-items-center rounded-xl border border-white/10 bg-white/5 text-sky-100 ring-1 ring-white/5">
            {icon}
          </div>
        ) : null}
        <div className="text-lg font-semibold tracking-tight text-slate-100">
          {title}
        </div>
      </div>
      {right ? <div className="flex items-center gap-2">{right}</div> : null}
    </div>
  );
}

export default function ScenarioAnalyzerUI() {
  type TabKey = "Portfolio" | "Simulation" | "Market Analytics";

  const [activeTab, setActiveTab] = useState<TabKey>("Portfolio");

  const [simulationPresetMarketId, setSimulationPresetMarketId] =
    useState<string | null>(null);

  const [portfolioIframeHeight, setPortfolioIframeHeight] =
    useState<number>(900);
  const portfolioDashboardUrl = DEFAULT_PORTFOLIO_DASHBOARD_URL;

  useEffect(() => {
    let allowedOrigin: string | null = null;
    try {
      allowedOrigin = new URL(portfolioDashboardUrl).origin;
    } catch {
      allowedOrigin = null;
    }

    const onMessage = (event: MessageEvent) => {
      if (allowedOrigin && event.origin !== allowedOrigin) return;
      const data = event.data as unknown;
      if (!data || typeof data !== "object") return;
      const msg = data as { type?: unknown; height?: unknown };
      if (msg.type !== "POLYWATCH_PORTFOLIO_HEIGHT") return;
      if (typeof msg.height !== "number" || !Number.isFinite(msg.height)) return;

      const next = Math.max(300, Math.min(6000, Math.round(msg.height)));
      setPortfolioIframeHeight(next);
    };

    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [portfolioDashboardUrl]);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      {/* Vapor-trail background (static) */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        {/* base glows */}
        <div className="absolute left-1/3 top-[-12%] h-[540px] w-[540px] rounded-full bg-sky-500/10 blur-[90px]" />
        <div className="absolute right-[-12%] top-1/4 h-[560px] w-[560px] rounded-full bg-violet-500/10 blur-[100px]" />
        <div className="absolute left-[-12%] bottom-[-12%] h-[560px] w-[560px] rounded-full bg-indigo-500/10 blur-[110px]" />
        <div className="absolute left-[10%] top-[30%] h-[420px] w-[420px] rounded-full bg-indigo-500/8 blur-[110px]" />

        {/* trails */}
        <div className="absolute left-[-20%] top-[18%] h-40 w-[1100px] rotate-[-12deg] rounded-full bg-gradient-to-r from-sky-500/0 via-sky-500/20 to-violet-500/0 blur-3xl mix-blend-screen" />
        <div className="absolute right-[-20%] top-[44%] h-40 w-[1100px] rotate-[10deg] rounded-full bg-gradient-to-r from-indigo-500/0 via-violet-500/22 to-sky-500/0 blur-3xl mix-blend-screen" />
        <div className="absolute left-[-25%] top-[68%] h-36 w-[1000px] rotate-[-6deg] rounded-full bg-gradient-to-r from-violet-500/0 via-indigo-500/20 to-sky-500/0 blur-3xl mix-blend-screen" />
        <div className="absolute left-[-30%] top-[8%] h-32 w-[980px] rotate-[14deg] rounded-full bg-gradient-to-r from-indigo-500/0 via-indigo-500/22 to-sky-500/0 blur-3xl mix-blend-screen" />
        <div className="absolute right-[-28%] top-[78%] h-32 w-[980px] rotate-[-10deg] rounded-full bg-gradient-to-r from-sky-500/0 via-violet-500/20 to-indigo-500/0 blur-3xl mix-blend-screen" />
      </div>

      <div className="relative mx-auto max-w-6xl px-5 pb-10 pt-7">
        {/* Top bar */}
        <div className="relative flex items-center justify-center">
          {/* right actions */}
          <div className="absolute right-0 top-0 flex items-center gap-3">
            <button className="grid h-11 w-11 place-items-center rounded-2xl border border-white/10 bg-white/5 shadow-xl shadow-black/20 ring-1 ring-white/5 hover:bg-white/10">
              <Settings className="h-5 w-5 text-slate-200" />
            </button>
          </div>

          <div className="py-2 text-center">
            <div className="text-4xl font-extrabold tracking-tight">
              <span className="bg-gradient-to-r from-sky-200 via-indigo-200 to-violet-200 bg-clip-text text-transparent drop-shadow-[0_0_18px_rgba(129,140,248,0.25)]">
                Polywatch
              </span>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="mt-6 flex items-center gap-2 border-b border-white/10">
          {(
            [
              {
                key: "Portfolio" as const,
                icon: <PieChart className="h-4 w-4" />,
              },
              {
                key: "Simulation" as const,
                icon: <GitBranch className="h-4 w-4" />,
              },
              {
                key: "Market Analytics" as const,
                icon: <BarChart3 className="h-4 w-4" />,
              },
            ] satisfies Array<{ key: TabKey; icon: React.ReactNode }>
          ).map((t) => {
            const on = activeTab === t.key;
            return (
              <button
                key={t.key}
                onClick={() => setActiveTab(t.key)}
                className={`relative -mb-px inline-flex items-center gap-2 rounded-t-xl px-4 py-3 text-sm font-semibold transition-colors ${
                  on
                    ? "border border-white/10 border-b-slate-900/0 bg-white/10 text-sky-100"
                    : "text-slate-300/80 hover:text-slate-200"
                }`}
              >
                <span
                  className={`${on ? "text-sky-200" : "text-slate-300/80"}`}
                >
                  {t.icon}
                </span>
                {t.key}
                {on ? (
                  <span className="absolute inset-x-3 bottom-0 h-[2px] rounded-full bg-gradient-to-r from-sky-400/70 via-indigo-400/60 to-violet-400/60" />
                ) : null}
              </button>
            );
          })}
        </div>

        {/* Main grid */}
        <div className="mt-6">
          <div className={activeTab === "Simulation" ? "" : "hidden"}>
            <ScenarioTester presetMarketId={simulationPresetMarketId} />
          </div>

          {activeTab === "Portfolio" && (
            <Card>
              <CardHeader title="Portfolio" icon={<PieChart className="h-5 w-5" />} />
              <div className="px-5 py-4">
                <div className="rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
                  {portfolioDashboardUrl ? (
                    <iframe
                      title="Portfolio Dashboard"
                      src={portfolioDashboardUrl}
                      className="w-full"
                      style={{ height: portfolioIframeHeight }}
                      sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
                    />
                  ) : (
                    <div className="p-6 text-sm text-slate-300/70">
                      Portfolio dashboard is not configured.
                    </div>
                  )}
                </div>
              </div>
            </Card>
          )}

          {activeTab === "Market Analytics" && (
              <div className="grid grid-cols-12 gap-5">
                <Card className="col-span-12">
                  <CardHeader
                    title="Market Analytics"
                    icon={<BarChart3 className="h-5 w-5" />}
                  />

                  <div className="px-5 py-4">
                    <AlertsAnalytics
                      onRunCorrelationSimulation={(marketId) => {
                        setSimulationPresetMarketId(marketId);
                        setActiveTab("Simulation");
                        window.requestAnimationFrame(() => {
                          window.scrollTo({ top: 0, behavior: "smooth" });
                        });
                      }}
                    />
                  </div>
                </Card>
              </div>
            )}
        </div>
      </div>
    </div>
  );
}
