"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ENDPOINTS } from "@/lib/api-config"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Cell,
} from "recharts"
import { ErrorBoundary } from "react-error-boundary"
import { ChevronDown, ChevronUp } from "lucide-react"

// Constants
const DEFAULT_PLACEHOLDER_IMAGE = '/placeholder-fighter.png'
const DEFAULT_VALUE = "0"
const DEFAULT_PERCENTAGE = "0%"
const UNRANKED_VALUE = 99

// Type definitions
type FightResult = 'win' | 'loss' | 'draw' | 'nc'

interface FighterDetailsProps {
  fighterName: string
}

interface FighterStats {
  name: string;
  image_url?: string;
  record: string;
  height: string;
  weight: string;
  reach: string;
  stance: string;
  dob: string;
  slpm: string;
  str_acc: string;
  sapm: string;
  str_def: string;
  td_avg: string;
  td_acc: string;
  td_def: string;
  sub_avg: string;
  weight_class?: string;
  nickname?: string;
  last_5_fights?: FightHistory[];
  ranking?: string | number;
  tap_link?: string;
}

interface FightHistory {
  opponent_name: string;
  opponent_display_name?: string;
  result: string;
  method: string;
  round: number;
  time: string;
  event: string;
  date: string;
  opponent_stats?: FighterStats;
  kd: string;
  sig_str: string;
  sig_str_pct: string;
  total_str: string;
  head_str: string;
  body_str: string;
  leg_str: string;
  takedowns: string;
  td_pct: string;
  ctrl: string;
}

interface ChartData {
  name: string;
  value: number;
  color: string;
}

interface ChartDataSet {
  strikeData: ChartData[];
  strikeAccuracyData: ChartData[];
  grappleData: ChartData[];
  grappleAccuracyData: ChartData[];
  overallStats: { subject: string; A: number }[];
  strikeDistribution: { name: string; value: number; percentage: number }[];
}

interface FightStat {
  label: string;
  value: string;
}

interface FightStatCategory {
  category: string;
  stats: FightStat[];
}

// Error Fallback Component
function ChartErrorFallback({ error }: { error: Error }) {
  return (
    <div className="p-4 text-sm text-red-500">
      Failed to load chart: {error.message}
    </div>
  )
}

// Result color mapping
const RESULT_COLORS: Record<FightResult, string> = {
  win: 'text-green-500',
  loss: 'text-red-500',
  draw: 'text-yellow-500',
  nc: 'text-gray-500'
}

// Simplified chart data calculations
function calculateChartData(stats: FighterStats | null): ChartDataSet {
  if (!stats) return {
    strikeData: [],
    strikeAccuracyData: [],
    grappleData: [],
    grappleAccuracyData: [],
    overallStats: [],
    strikeDistribution: []
  }

  return {
    strikeData: [
      { name: 'Strikes Landed/min', value: parseFloat(stats.slpm) || 0, color: '#3b82f6' },
      { name: 'Strikes Absorbed/min', value: parseFloat(stats.sapm) || 0, color: '#3b82f6' },
    ],
    strikeAccuracyData: [
      { name: 'Strike Accuracy', value: parseFloat(stats.str_acc) || 0, color: '#3b82f6' },
      { name: 'Strike Defense', value: parseFloat(stats.str_def) || 0, color: '#3b82f6' },
    ],
    grappleData: [
      { name: 'Takedowns/15min', value: parseFloat(stats.td_avg) || 0, color: '#3b82f6' },
      { name: 'Submissions/15min', value: parseFloat(stats.sub_avg) || 0, color: '#3b82f6' },
    ],
    grappleAccuracyData: [
      { name: 'Takedown Accuracy', value: parseFloat(stats.td_acc) || 0, color: '#3b82f6' },
      { name: 'Takedown Defense', value: parseFloat(stats.td_def) || 0, color: '#3b82f6' },
    ],
    overallStats: [
      { subject: 'Strike Power', A: (parseFloat(stats.slpm) / 10) * 100 || 0 },
      { subject: 'Strike Defense', A: parseFloat(stats.str_def) || 0 },
      { subject: 'Grappling', A: (parseFloat(stats.td_avg) / 5) * 100 || 0 },
      { subject: 'Submission', A: (parseFloat(stats.sub_avg) / 2) * 100 || 0 },
      { subject: 'Accuracy', A: parseFloat(stats.str_acc) || 0 },
    ],
    strikeDistribution: (() => {
      const totalStrikes = parseFloat(stats.slpm) + parseFloat(stats.sapm);
      return [
        { name: 'Strikes Landed', value: parseFloat(stats.slpm), percentage: (parseFloat(stats.slpm) / totalStrikes) * 100 },
        { name: 'Strikes Absorbed', value: parseFloat(stats.sapm), percentage: (parseFloat(stats.sapm) / totalStrikes) * 100 },
      ];
    })()
  }
}

export function FighterDetails({ fighterName }: FighterDetailsProps) {
  const [stats, setStats] = React.useState<FighterStats | null>(null)
  const [fightHistory, setFightHistory] = React.useState<FightHistory[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [imageError, setImageError] = React.useState(false)
  const [expandedFight, setExpandedFight] = React.useState<number | null>(null)

  // Fetch fighter data
  React.useEffect(() => {
    const fetchFighterData = async () => {
      if (!fighterName || fighterName === 'undefined') {
        setError('Invalid fighter name');
        setIsLoading(false);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch(ENDPOINTS.FIGHTER(fighterName));
        if (!response.ok) {
          throw new Error(response.status === 404 ? 'Fighter not found' : 'Failed to fetch fighter data');
        }

        const data = await response.json();
        setStats(data);
        setFightHistory(data.last_5_fights || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch fighter data');
      } finally {
        setIsLoading(false);
      }
    };

    fetchFighterData();
  }, [fighterName]);

  // Calculate all chart data at once
  const chartData = React.useMemo(() => calculateChartData(stats), [stats]);

  // Calculate fight stats
  const fightStats = React.useMemo<FightStatCategory[]>(() => {
    if (!stats) return [];
    return [
      { 
        category: 'Striking', 
        stats: [
          { label: 'Strikes Landed/min', value: parseFloat(stats.slpm).toFixed(1) },
          { label: 'Strike Accuracy', value: stats.str_acc.replace('%', '') + '%' },
          { label: 'Strikes Absorbed/min', value: parseFloat(stats.sapm).toFixed(1) },
          { label: 'Strike Defense', value: stats.str_def.replace('%', '') + '%' },
        ]
      },
      { 
        category: 'Grappling', 
        stats: [
          { label: 'Takedowns/15min', value: parseFloat(stats.td_avg).toFixed(1) },
          { label: 'Takedown Accuracy', value: stats.td_acc.replace('%', '') + '%' },
          { label: 'Takedown Defense', value: stats.td_def.replace('%', '') + '%' },
          { label: 'Submissions/15min', value: parseFloat(stats.sub_avg).toFixed(1) },
        ]
      },
    ];
  }, [stats]);

  const calculateAge = (dob: string) => {
    const birthDate = new Date(dob);
    const today = new Date();
    let age = today.getFullYear() - birthDate.getFullYear();
    const monthDiff = today.getMonth() - birthDate.getMonth();
    
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
      age--;
    }
    
    return age;
  };

  if (isLoading) {
    return <div className="p-8 text-center animate-pulse">Loading fighter data...</div>
  }

  if (error) {
    return (
      <div className="p-8 text-center text-destructive border border-destructive rounded-lg">
        {error}
      </div>
    )
  }

  if (!stats) {
    return <div className="p-8 text-center">No fighter data available</div>
  }

  const getResultColor = (result: string) => {
    return RESULT_COLORS[result.toLowerCase() as FightResult] || RESULT_COLORS.nc
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-5xl animate-in fade-in duration-700">
      <div className="space-y-8 flex flex-col items-center">
        {/* Fighter Header */}
        <div className="w-full grid grid-cols-1 md:grid-cols-3 gap-8 bg-accent/50 p-6 rounded-lg transition-all duration-300 hover:bg-accent/60">
          <div className="relative aspect-square md:col-span-1">
            {!imageError ? (
              stats.tap_link ? (
                <a 
                  href={stats.tap_link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block w-full h-full transform transition-all duration-500 hover:scale-105 hover:shadow-xl rounded-lg overflow-hidden"
                >
                  <img 
                    src={stats.image_url || DEFAULT_PLACEHOLDER_IMAGE} 
                    alt={stats.name}
                    className="w-full h-full object-cover object-top transition-transform duration-500"
                    onError={() => setImageError(true)}
                  />
                  <div className="absolute inset-0 bg-black/0 hover:bg-black/20 transition-colors duration-300 flex items-center justify-center">
                    <span className="text-white opacity-0 hover:opacity-100 transition-all duration-300 transform translate-y-2 hover:translate-y-0">View on Tapology</span>
                  </div>
                </a>
              ) : (
                <div className="w-full h-full rounded-lg overflow-hidden transition-transform duration-300 hover:scale-102">
                  <img 
                    src={stats.image_url || DEFAULT_PLACEHOLDER_IMAGE} 
                    alt={stats.name}
                    className="w-full h-full object-cover object-top transition-transform duration-300"
                    onError={() => setImageError(true)}
                  />
                </div>
              )
            ) : (
              <div className="w-full h-full bg-gray-200 rounded-lg flex items-center justify-center transition-colors duration-300">
                <span className="text-gray-500">No image available</span>
              </div>
            )}
          </div>
          <div className="md:col-span-2">
            <div className="space-y-4">
              <div className="animate-in slide-in-from-right duration-700 delay-200">
                <h2 className="text-4xl font-bold">{stats.name}</h2>
                {stats.nickname && (
                  <p className="text-xl text-muted-foreground">"{stats.nickname}"</p>
                )}
                <p className="text-2xl font-semibold mt-2">{stats.record}</p>
                {stats.weight_class && (
                  <p className="text-lg text-muted-foreground">{stats.weight_class}</p>
                )}
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6 animate-in slide-in-from-bottom duration-700 delay-300">
                {[
                  { label: 'Height', value: stats.height },
                  { label: 'Weight', value: stats.weight },
                  { label: 'Reach', value: stats.reach },
                  { label: 'Stance', value: stats.stance },
                  { label: 'DOB', value: `${stats.dob} (${calculateAge(stats.dob)} years)` },
                  { label: 'Ranking', value: stats.ranking === 99 || stats.ranking === '99' ? 'Unranked' : `#${stats.ranking}` },
                ].map(({ label, value }, index) => (
                  <div 
                    key={label} 
                    className="bg-background/50 p-3 rounded-lg transition-all duration-300 hover:bg-background/70 hover:shadow-md"
                    style={{ animationDelay: `${(index + 1) * 100}ms` }}
                  >
                    <p className="text-sm text-muted-foreground">{label}</p>
                    <p className="font-medium">{value || 'N/A'}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Stats and History Tabs */}
        <div className="w-full">
          <Tabs defaultValue="overview" className="w-full">
            <TabsList className="grid w-full grid-cols-3 animate-in fade-in duration-700 delay-500">
              <TabsTrigger value="overview" className="transition-all duration-300 data-[state=active]:animate-in data-[state=active]:zoom-in-95">Overview</TabsTrigger>
              <TabsTrigger value="stats" className="transition-all duration-300 data-[state=active]:animate-in data-[state=active]:zoom-in-95">Detailed Stats</TabsTrigger>
              <TabsTrigger value="history" className="transition-all duration-300 data-[state=active]:animate-in data-[state=active]:zoom-in-95">Fight History</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="animate-in slide-in-from-bottom duration-500">
              <div className="grid gap-6">
                {/* Fighter Overview Card */}
                <Card>
                  <CardHeader>
                    <CardTitle>Fighter Overview</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      {/* Radar Chart */}
                      <div className="h-[400px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <RadarChart data={chartData.overallStats}>
                            <PolarGrid strokeDasharray="3 3" />
                            <PolarAngleAxis 
                              dataKey="subject" 
                              tick={{ fill: 'currentColor', fontSize: 12 }}
                            />
                            <PolarRadiusAxis angle={30} domain={[0, 100]} />
                            <Radar
                              name={stats.name}
                              dataKey="A"
                              stroke="#3b82f6"
                              fill="#3b82f6"
                              fillOpacity={0.6}
                            />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>

                      {/* Quick Stats */}
                      <div className="space-y-6">
                        <h3 className="text-lg font-semibold">Quick Stats</h3>
                        <div className="grid grid-cols-2 gap-4">
                          {fightStats.map((category) => (
                            <div key={category.category} className="space-y-3">
                              <h4 className="text-sm font-medium text-muted-foreground">{category.category}</h4>
                              {category.stats.map((stat) => (
                                <div key={stat.label} className="flex justify-between items-center">
                                  <span className="text-sm">{stat.label}</span>
                                  <span className="font-semibold">{stat.value}</span>
                                </div>
                              ))}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Strike Distribution Card */}
                <Card>
                  <CardHeader>
                    <CardTitle>Strike Distribution</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[200px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData.strikeDistribution} layout="vertical">
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis type="number" domain={[0, 'dataMax + 1']} />
                          <YAxis dataKey="name" type="category" />
                          <Tooltip
                            content={({ payload, label }) => {
                              if (payload && payload.length && payload[0].value != null) {
                                const value = Number(payload[0].value);
                                const percentage = payload[0].payload.percentage;
                                return (
                                  <div className="bg-background/95 p-2 rounded-lg border shadow-sm">
                                    <p className="font-medium">{label}</p>
                                    <p className="text-sm">{`${value.toFixed(1)} strikes/min (${percentage.toFixed(1)}%)`}</p>
                                  </div>
                                );
                              }
                              return null;
                            }}
                          />
                          <Bar dataKey="value" fill="#3b82f6">
                            {chartData.strikeDistribution.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={index === 0 ? '#3b82f6' : '#3b82f6'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="stats" className="space-y-6 animate-in slide-in-from-bottom duration-500">
              {/* Striking Stats */}
              <ErrorBoundary FallbackComponent={ChartErrorFallback}>
                <Card>
                  <CardHeader>
                    <CardTitle>Striking Statistics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4 mb-6">
                      <div className="space-y-2 p-4 bg-accent/10 rounded-lg">
                        <p className="text-sm text-muted-foreground">Strikes Landed per min</p>
                        <p className="text-3xl font-bold">{parseFloat(stats.slpm).toFixed(1)}</p>
                        <p className="text-sm text-muted-foreground">Striking Output</p>
                      </div>
                      <div className="space-y-2 p-4 bg-accent/10 rounded-lg">
                        <p className="text-sm text-muted-foreground">Strike Accuracy</p>
                        <p className="text-3xl font-bold">{stats.str_acc}</p>
                        <p className="text-sm text-muted-foreground">Strike Success Rate</p>
                      </div>
                      <div className="space-y-2 p-4 bg-accent/10 rounded-lg">
                        <p className="text-sm text-muted-foreground">Strikes Absorbed per min</p>
                        <p className="text-3xl font-bold">{parseFloat(stats.sapm).toFixed(1)}</p>
                        <p className="text-sm text-muted-foreground">Strikes Received</p>
                      </div>
                      <div className="space-y-2 p-4 bg-accent/10 rounded-lg">
                        <p className="text-sm text-muted-foreground">Strike Defense</p>
                        <p className="text-3xl font-bold">{stats.str_def}</p>
                        <p className="text-sm text-muted-foreground">Strike Evasion Rate</p>
                      </div>
                    </div>
                    <div className="space-y-6">
                      {/* Strike Output Chart */}
                      <div className="h-[200px]">
                        <h4 className="text-sm font-medium mb-2">Strike Output</h4>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={chartData.strikeData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                            <XAxis 
                              dataKey="name" 
                              tick={{ fill: 'currentColor', fontSize: 12 }}
                              axisLine={{ stroke: 'currentColor', opacity: 0.2 }}
                            />
                            <YAxis 
                              domain={[0, 'dataMax + 2']}
                              tick={{ fill: 'currentColor', fontSize: 12 }}
                              axisLine={{ stroke: 'currentColor', opacity: 0.2 }}
                            />
                            <Tooltip
                              content={({ payload, label }) => {
                                if (payload && payload.length && payload[0].value != null) {
                                  const value = Number(payload[0].value);
                                  return (
                                    <div className="bg-background/95 p-2 rounded-lg border shadow-sm">
                                      <p className="font-medium">{label}</p>
                                      <p className="text-sm">{`${value.toFixed(1)} strikes/min`}</p>
                                    </div>
                                  );
                                }
                                return null;
                              }}
                            />
                            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                              {chartData.strikeData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      {/* Strike Accuracy Chart */}
                      <div className="h-[200px]">
                        <h4 className="text-sm font-medium mb-2">Strike Accuracy</h4>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={chartData.strikeAccuracyData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                            <XAxis 
                              dataKey="name" 
                              tick={{ fill: 'currentColor', fontSize: 12 }}
                              axisLine={{ stroke: 'currentColor', opacity: 0.2 }}
                            />
                            <YAxis 
                              domain={[0, 100]}
                              tick={{ fill: 'currentColor', fontSize: 12 }}
                              axisLine={{ stroke: 'currentColor', opacity: 0.2 }}
                              tickFormatter={(value) => `${value}%`}
                            />
                            <Tooltip
                              content={({ payload, label }) => {
                                if (payload && payload.length && payload[0].value != null) {
                                  const value = Number(payload[0].value);
                                  return (
                                    <div className="bg-background/95 p-2 rounded-lg border shadow-sm">
                                      <p className="font-medium">{label}</p>
                                      <p className="text-sm">{`${value}%`}</p>
                                    </div>
                                  );
                                }
                                return null;
                              }}
                            />
                            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                              {chartData.strikeAccuracyData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </ErrorBoundary>

              {/* Grappling Stats */}
              <ErrorBoundary FallbackComponent={ChartErrorFallback}>
                <Card>
                  <CardHeader>
                    <CardTitle>Grappling Statistics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4 mb-6">
                      <div className="space-y-2 p-4 bg-accent/10 rounded-lg">
                        <p className="text-sm text-muted-foreground">Takedowns per 15 min</p>
                        <p className="text-3xl font-bold">{parseFloat(stats.td_avg).toFixed(1)}</p>
                        <p className="text-sm text-muted-foreground">Grappling Frequency</p>
                      </div>
                      <div className="space-y-2 p-4 bg-accent/10 rounded-lg">
                        <p className="text-sm text-muted-foreground">Takedown Accuracy</p>
                        <p className="text-3xl font-bold">{stats.td_acc}</p>
                        <p className="text-sm text-muted-foreground">Takedown Success Rate</p>
                      </div>
                      <div className="space-y-2 p-4 bg-accent/10 rounded-lg">
                        <p className="text-sm text-muted-foreground">Takedown Defense</p>
                        <p className="text-3xl font-bold">{stats.td_def}</p>
                        <p className="text-sm text-muted-foreground">Takedown Prevention</p>
                      </div>
                      <div className="space-y-2 p-4 bg-accent/10 rounded-lg">
                        <p className="text-sm text-muted-foreground">Submissions per 15 min</p>
                        <p className="text-3xl font-bold">{parseFloat(stats.sub_avg).toFixed(1)}</p>
                        <p className="text-sm text-muted-foreground">Submission Threat</p>
                      </div>
                    </div>
                    <div className="space-y-6">
                      {/* Grappling Output Chart */}
                      <div className="h-[200px]">
                        <h4 className="text-sm font-medium mb-2">Grappling Output</h4>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={chartData.grappleData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                            <XAxis 
                              dataKey="name" 
                              tick={{ fill: 'currentColor', fontSize: 12 }}
                              axisLine={{ stroke: 'currentColor', opacity: 0.2 }}
                            />
                            <YAxis 
                              domain={[0, 5]}
                              tick={{ fill: 'currentColor', fontSize: 12 }}
                              axisLine={{ stroke: 'currentColor', opacity: 0.2 }}
                            />
                            <Tooltip
                              content={({ payload, label }) => {
                                if (payload && payload.length && payload[0].value != null) {
                                  const value = Number(payload[0].value);
                                  return (
                                    <div className="bg-background/95 p-2 rounded-lg border shadow-sm">
                                      <p className="font-medium">{label}</p>
                                      <p className="text-sm">{`${value.toFixed(1)} per 15min`}</p>
                                    </div>
                                  );
                                }
                                return null;
                              }}
                            />
                            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                              {chartData.grappleData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      {/* Grappling Accuracy Chart */}
                      <div className="h-[200px]">
                        <h4 className="text-sm font-medium mb-2">Grappling Accuracy</h4>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={chartData.grappleAccuracyData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                            <XAxis 
                              dataKey="name" 
                              tick={{ fill: 'currentColor', fontSize: 12 }}
                              axisLine={{ stroke: 'currentColor', opacity: 0.2 }}
                            />
                            <YAxis 
                              domain={[0, 100]}
                              tick={{ fill: 'currentColor', fontSize: 12 }}
                              axisLine={{ stroke: 'currentColor', opacity: 0.2 }}
                              tickFormatter={(value) => `${value}%`}
                            />
                            <Tooltip
                              content={({ payload, label }) => {
                                if (payload && payload.length && payload[0].value != null) {
                                  const value = Number(payload[0].value);
                                  return (
                                    <div className="bg-background/95 p-2 rounded-lg border shadow-sm">
                                      <p className="font-medium">{label}</p>
                                      <p className="text-sm">{`${value}%`}</p>
                                    </div>
                                  );
                                }
                                return null;
                              }}
                            />
                            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                              {chartData.grappleAccuracyData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </ErrorBoundary>
            </TabsContent>

            <TabsContent value="history" className="animate-in slide-in-from-bottom duration-500">
              <Card>
                <CardHeader>
                  <CardTitle>Last 5 Fights</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    {fightHistory.length > 0 ? (
                      fightHistory.map((fight, index) => (
                        <div key={`${fight.opponent_name}-${fight.date}-${index}`} className="border rounded-lg overflow-hidden">
                          <div 
                            className="flex items-center justify-between p-4 bg-card hover:bg-accent/50 transition-colors cursor-pointer"
                            onClick={() => setExpandedFight(expandedFight === index ? null : index)}
                          >
                            <div className="space-y-1">
                              <div className="flex items-center gap-2">
                                <p className="font-semibold text-lg">vs. {fight.opponent_name}</p>
                                <span className={`px-2 py-0.5 rounded text-sm font-medium ${getResultColor(fight.result)}`}>
                                  {fight.result.toUpperCase()}
                                </span>
                              </div>
                              <p className="text-sm text-muted-foreground">{fight.event}</p>
                              <p className="text-sm text-muted-foreground">{fight.date}</p>
                            </div>
                            <div className="text-right flex items-center gap-2">
                              <div>
                                <p className="font-medium">{fight.method}</p>
                                {(fight.round || fight.time) && (
                                  <p className="text-sm text-muted-foreground">
                                    Round {fight.round} - {fight.time}
                                  </p>
                                )}
                              </div>
                              {expandedFight === index ? (
                                <ChevronUp className="h-5 w-5" />
                              ) : (
                                <ChevronDown className="h-5 w-5" />
                              )}
                            </div>
                          </div>
                          
                          {/* Expanded fight details */}
                          {expandedFight === index && (
                            <div className="border-t">
                              {/* Fight Overview */}
                              <div className="grid grid-cols-2 gap-4 p-6 bg-accent/5">
                                <div className="p-4 bg-background rounded-lg text-center shadow-sm hover:shadow-md transition-shadow">
                                  <p className="text-4xl font-bold text-primary">{fight.kd || '0'}</p>
                                  <p className="text-sm font-medium text-muted-foreground mt-1">Knockdowns</p>
                                </div>
                                <div className="p-4 bg-background rounded-lg text-center shadow-sm hover:shadow-md transition-shadow">
                                  <p className="text-4xl font-bold text-primary">{fight.ctrl || '0:00'}</p>
                                  <p className="text-sm font-medium text-muted-foreground mt-1">Control Time</p>
                                </div>
                              </div>

                              {/* Stats Grid */}
                              <div className="p-6 space-y-8">
                                {/* Total and Significant Strikes */}
                                <div className="grid grid-cols-2 gap-6">
                                  {/* Total Strikes */}
                                  <div className="p-4 bg-accent/5 rounded-lg">
                                    <h4 className="text-sm font-medium text-muted-foreground mb-3">Total Strikes</h4>
                                    <p className="text-3xl font-bold">{fight.total_str}</p>
                                  </div>

                                  {/* Significant Strikes */}
                                  <div className="p-4 bg-accent/5 rounded-lg">
                                    <h4 className="text-sm font-medium text-muted-foreground mb-3">Significant Strikes</h4>
                                    <div className="space-y-1">
                                      <p className="text-3xl font-bold">{fight.sig_str}</p>
                                      <p className="text-sm font-medium text-primary">{fight.sig_str_pct}</p>
                                    </div>
                                  </div>
                                </div>

                                {/* Strike Distribution */}
                                <div className="bg-accent/5 rounded-lg p-4">
                                  <h4 className="text-sm font-medium text-muted-foreground mb-4">Strike Distribution</h4>
                                  <div className="grid grid-cols-3 gap-6">
                                    <div className="text-center p-3 bg-background rounded-lg shadow-sm">
                                      <h4 className="text-xs font-medium text-muted-foreground mb-2">Head</h4>
                                      <p className="text-2xl font-bold">{fight.head_str}</p>
                                    </div>
                                    <div className="text-center p-3 bg-background rounded-lg shadow-sm">
                                      <h4 className="text-xs font-medium text-muted-foreground mb-2">Body</h4>
                                      <p className="text-2xl font-bold">{fight.body_str}</p>
                                    </div>
                                    <div className="text-center p-3 bg-background rounded-lg shadow-sm">
                                      <h4 className="text-xs font-medium text-muted-foreground mb-2">Leg</h4>
                                      <p className="text-2xl font-bold">{fight.leg_str}</p>
                                    </div>
                                  </div>
                                </div>

                                {/* Takedowns */}
                                <div className="bg-accent/5 rounded-lg p-4">
                                  <h4 className="text-sm font-medium text-muted-foreground mb-3">Takedowns</h4>
                                  <div className="grid grid-cols-2 gap-6">
                                    <div className="p-3 bg-background rounded-lg shadow-sm text-center">
                                      <p className="text-2xl font-bold">{fight.takedowns}</p>
                                      <p className="text-xs font-medium text-muted-foreground mt-1">Attempts</p>
                                    </div>
                                    <div className="p-3 bg-background rounded-lg shadow-sm text-center">
                                      <p className="text-2xl font-bold">{fight.td_pct}</p>
                                      <p className="text-xs font-medium text-muted-foreground mt-1">Success Rate</p>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      ))
                    ) : (
                      <p className="text-center text-muted-foreground">
                        No fight history available
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  )
} 