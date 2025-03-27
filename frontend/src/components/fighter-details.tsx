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

  // Fetch fighter data and fight history
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
        // Log the API call for debugging
        console.log(`Fetching fighter data for: ${fighterName}`);
        
        const response = await fetch(ENDPOINTS.FIGHTER(fighterName));
        if (!response.ok) {
          throw new Error(response.status === 404 ? 'Fighter not found' : 'Failed to fetch fighter data');
        }

        const data = await response.json();
        console.log("Raw fighter data received:", data);
        
        // Check for fight history in different possible formats
        let fightHistoryData: FightHistory[] = [];
        if (Array.isArray(data?.last_5_fights) && data.last_5_fights.length > 0) {
          console.log("Found last_5_fights in main response");
          fightHistoryData = data.last_5_fights;
        } else if (Array.isArray(data?.fights) && data.fights.length > 0) {
          console.log("Found fights in main response");
          fightHistoryData = data.fights;
        } else if (data?.fight_history && Array.isArray(data.fight_history) && data.fight_history.length > 0) {
          console.log("Found fight_history in main response");
          fightHistoryData = data.fight_history;
        }
        
        // Map and sanitize fight history data
        const processedFightHistory = fightHistoryData.map((fight: any) => {
          return {
            opponent_name: fight?.opponent_name || fight?.opponent || 'Unknown Opponent',
            opponent_display_name: fight?.opponent_display_name || fight?.opponent || 'Unknown Opponent',
            result: fight?.result || 'NC',
            method: fight?.method || 'N/A',
            round: fight?.round || 0,
            time: fight?.time || '0:00',
            event: fight?.event || 'Unknown Event',
            date: fight?.date || 'Unknown Date',
            opponent_stats: fight?.opponent_stats || undefined,
            kd: fight?.kd || '0',
            sig_str: fight?.sig_str || '0/0',
            sig_str_pct: fight?.sig_str_pct || '0%',
            total_str: fight?.total_str || '0/0',
            head_str: fight?.head_str || '0/0',
            body_str: fight?.body_str || '0/0',
            leg_str: fight?.leg_str || '0/0',
            takedowns: fight?.takedowns || '0/0',
            td_pct: fight?.td_pct || '0%',
            ctrl: fight?.ctrl || '0:00',
          };
        });
        
        console.log("Processed fight history:", processedFightHistory);
        
        // Properly map API fields to our expected structure
        const sanitizedData: Record<string, any> = {
          name: data?.fighter_name || data?.name || fighterName || '',
          image_url: data?.image_url || DEFAULT_PLACEHOLDER_IMAGE,
          record: data?.Record || data?.record || DEFAULT_VALUE,
          height: data?.Height || data?.height || DEFAULT_VALUE,
          weight: data?.Weight || data?.weight || DEFAULT_VALUE,
          reach: data?.Reach || data?.reach || DEFAULT_VALUE,
          stance: data?.STANCE || data?.stance || DEFAULT_VALUE,
          dob: data?.DOB || data?.dob || '',
          slpm: data?.SLpM || data?.SLPM || data?.slpm || DEFAULT_VALUE,
          str_acc: data?.['Str. Acc.'] || data?.str_acc || DEFAULT_PERCENTAGE,
          sapm: data?.SApM || data?.SAPM || data?.sapm || DEFAULT_VALUE,
          str_def: data?.['Str. Def'] || data?.str_def || DEFAULT_PERCENTAGE,
          td_avg: data?.['TD Avg.'] || data?.td_avg || DEFAULT_VALUE,
          td_acc: data?.['TD Acc.'] || data?.td_acc || DEFAULT_PERCENTAGE,
          td_def: data?.['TD Def.'] || data?.td_def || DEFAULT_PERCENTAGE,
          sub_avg: data?.['Sub. Avg.'] || data?.sub_avg || DEFAULT_VALUE,
          weight_class: data?.weight_class || '',
          nickname: data?.nickname || '',
          last_5_fights: processedFightHistory, // Store the processed fight history
          ranking: data?.ranking || UNRANKED_VALUE,
          tap_link: data?.tap_link || '',
        };
        
        console.log("Mapped fighter data:", sanitizedData);
        
        // Ensure string values for all fields that might be used with string methods
        Object.keys(sanitizedData).forEach(key => {
          if (key !== 'last_5_fights' && key !== 'ranking' && sanitizedData[key] === null) {
            sanitizedData[key] = typeof sanitizedData[key] === 'number' ? String(sanitizedData[key]) : '';
          }
        });
        
        setStats(sanitizedData as FighterStats);
        setFightHistory(processedFightHistory);
      } catch (err) {
        console.error('Error fetching fighter:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch fighter data');
        // Set default empty stats to avoid UI crashes
        setStats({
          name: fighterName || '',
          image_url: DEFAULT_PLACEHOLDER_IMAGE,
          record: DEFAULT_VALUE,
          height: DEFAULT_VALUE,
          weight: DEFAULT_VALUE,
          reach: DEFAULT_VALUE,
          stance: DEFAULT_VALUE,
          dob: '',
          slpm: DEFAULT_VALUE,
          str_acc: DEFAULT_PERCENTAGE,
          sapm: DEFAULT_VALUE,
          str_def: DEFAULT_PERCENTAGE,
          td_avg: DEFAULT_VALUE,
          td_acc: DEFAULT_PERCENTAGE,
          td_def: DEFAULT_PERCENTAGE,
          sub_avg: DEFAULT_VALUE,
        });
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
    
    const safeParseFloat = (value: string) => {
      try {
        if (!value || typeof value !== 'string') return 0;
        // Remove any non-numeric characters except decimal point
        const sanitized = value.replace(/[^\d.]/g, '');
        const num = parseFloat(sanitized);
        return isNaN(num) ? 0 : num;
      } catch (e) {
        console.error(`Error parsing value: ${value}`, e);
        return 0;
      }
    };
    
    const safeReplacePercent = (value: string) => {
      try {
        if (!value || typeof value !== 'string') return '0%';
        // Remove any existing % symbol and add it back
        const sanitized = value.replace(/%/g, '').trim();
        return `${sanitized}%`;
      } catch (e) {
        console.error(`Error processing percent: ${value}`, e);
        return '0%';
      }
    };
    
    return [
      { 
        category: 'Striking', 
        stats: [
          { label: 'Strikes Landed/min', value: safeParseFloat(stats.slpm).toFixed(1) },
          { label: 'Strike Accuracy', value: safeReplacePercent(stats.str_acc) },
          { label: 'Strikes Absorbed/min', value: safeParseFloat(stats.sapm).toFixed(1) },
          { label: 'Strike Defense', value: safeReplacePercent(stats.str_def) },
        ]
      },
      { 
        category: 'Grappling', 
        stats: [
          { label: 'Takedowns/15min', value: safeParseFloat(stats.td_avg).toFixed(1) },
          { label: 'Takedown Accuracy', value: safeReplacePercent(stats.td_acc) },
          { label: 'Takedown Defense', value: safeReplacePercent(stats.td_def) },
          { label: 'Submissions/15min', value: safeParseFloat(stats.sub_avg).toFixed(1) },
        ]
      },
    ];
  }, [stats]);

  const calculateAge = (dob: string) => {
    // Return empty string if DOB is not provided
    if (!dob) return '';
    
    try {
      // Check if the date is in a valid format
      if (!/^\d{4}-\d{2}-\d{2}$/.test(dob) && 
          !/^\d{2}\/\d{2}\/\d{4}$/.test(dob) &&
          !/^\w+ \d{1,2}, \d{4}$/.test(dob)) {
        return '';
      }
      
      const birthDate = new Date(dob);
      // Check if the date is valid
      if (isNaN(birthDate.getTime())) {
        return '';
      }
      
      const today = new Date();
      let age = today.getFullYear() - birthDate.getFullYear();
      const monthDiff = today.getMonth() - birthDate.getMonth();
      
      if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
        age--;
      }
      
      // Ensure age is reasonable (between 18 and 50 for fighters)
      if (age < 18 || age > 50) {
        console.warn(`Calculated age ${age} from DOB ${dob} seems suspicious`);
        return '';
      }
      
      return age;
    } catch (error) {
      console.error(`Error calculating age from DOB: ${dob}`, error);
      return '';
    }
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

  // Fight history tab content
  const FightHistoryView = () => (
    <div className="space-y-8 py-4 animate-in slide-in-from-bottom duration-700">
      <h4 className="text-xl font-semibold">Last {fightHistory.length} Fights</h4>
      
      {fightHistory.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground">
          <p>No fight history available</p>
        </div>
      ) : (
        <div className="space-y-4">
          {fightHistory.map((fight, index) => (
            <Card key={`${fight.opponent_name}-${fight.date}-${index}`} className="overflow-hidden">
              <CardContent className="p-0">
                <div 
                  className={`p-4 cursor-pointer hover:bg-accent/10 transition-colors flex flex-col md:flex-row md:items-center justify-between gap-4 ${
                    expandedFight === index ? 'bg-accent/10' : ''
                  }`}
                  onClick={() => setExpandedFight(expandedFight === index ? null : index)}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-4 h-4 rounded-full ${
                      fight.result.toLowerCase() === 'win' ? 'bg-green-500' : 
                      fight.result.toLowerCase() === 'loss' ? 'bg-red-500' : 
                      'bg-yellow-500'
                    }`} />
                    <div>
                      <p className="font-medium">{fight.opponent_display_name || fight.opponent_name}</p>
                      <p className="text-sm text-muted-foreground">{fight.date}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-6">
                    <div>
                      <p className="text-sm">Method</p>
                      <p className="font-medium">{fight.method}</p>
                    </div>
                    <div>
                      <p className="text-sm">Round</p>
                      <p className="font-medium">{fight.round}</p>
                    </div>
                    <div>
                      <p className="text-sm">Time</p>
                      <p className="font-medium">{fight.time}</p>
                    </div>
                    <div>
                      <ChevronDown
                        className={`h-5 w-5 transition-transform ${
                          expandedFight === index ? 'rotate-180' : ''
                        }`}
                      />
                    </div>
                  </div>
                </div>
                
                {/* Expanded fight stats */}
                {expandedFight === index && (
                  <div className="px-4 py-6 border-t border-border bg-accent/5">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                      <div>
                        <p className="text-sm text-muted-foreground">Knockdowns</p>
                        <p className="font-medium">{fight.kd}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Sig. Strikes</p>
                        <p className="font-medium">{fight.sig_str} ({fight.sig_str_pct})</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Total Strikes</p>
                        <p className="font-medium">{fight.total_str}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Takedowns</p>
                        <p className="font-medium">{fight.takedowns} ({fight.td_pct})</p>
                      </div>
                    </div>
                    <div className="mt-4 border-t border-border/50 pt-4">
                      <h5 className="font-medium mb-2">Strike Distribution</h5>
                      <div className="grid grid-cols-3 gap-4">
                        <div>
                          <p className="text-sm text-muted-foreground">Head</p>
                          <p className="font-medium">{fight.head_str}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Body</p>
                          <p className="font-medium">{fight.body_str}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Leg</p>
                          <p className="font-medium">{fight.leg_str}</p>
                        </div>
                      </div>
                    </div>
                    {fight.ctrl && fight.ctrl !== '0:00' && (
                      <div className="mt-4 border-t border-border/50 pt-4">
                        <p className="text-sm text-muted-foreground">Control Time</p>
                        <p className="font-medium">{fight.ctrl}</p>
                      </div>
                    )}
                    <div className="mt-4 border-t border-border/50 pt-4 text-sm text-muted-foreground">
                      <p>{fight.event}</p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );

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
                  { label: 'Height', value: stats.height || 'N/A' },
                  { label: 'Weight', value: stats.weight || 'N/A' },
                  { label: 'Reach', value: stats.reach || 'N/A' },
                  { label: 'Stance', value: stats.stance || 'N/A' },
                  { label: 'DOB', value: stats.dob ? `${stats.dob} ${calculateAge(stats.dob) ? `(${calculateAge(stats.dob)} years)` : ''}` : 'N/A' },
                  { label: 'Ranking', value: stats.ranking === 99 || stats.ranking === '99' ? 'Unranked' : `#${stats.ranking || 'N/A'}` },
                ].map(({ label, value }, index) => (
                  <div 
                    key={label} 
                    className="bg-background/50 p-3 rounded-lg transition-all duration-300 hover:bg-background/70 hover:shadow-md"
                    style={{ animationDelay: `${(index + 1) * 100}ms` }}
                  >
                    <p className="text-sm text-muted-foreground">{label}</p>
                    <p className="font-medium">{value}</p>
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
              <FightHistoryView />
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  )
} 