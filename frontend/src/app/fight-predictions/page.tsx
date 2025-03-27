"use client"

import { useState, useEffect } from "react"
import { FighterSearch } from "@/components/fighter-search"
import { Card, CardContent } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { motion, AnimatePresence } from "framer-motion"
import { ArrowLeftRight, Swords, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useToast } from "@/components/ui/use-toast"
import { ENDPOINTS } from "@/lib/api-config"

interface FighterStats {
  name: string;
  image_url?: string;
  record: string;
  height: string;
  weight: string;
  reach: string;
  stance: string;
  slpm: string;
  str_acc: string;
  sapm: string;
  str_def: string;
  td_avg: string;
  td_acc: string;
  td_def: string;
  sub_avg: string;
  ranking?: string | number;
  tap_link?: string;
}

interface Prediction {
  winner: string;
  loser: string;
  winner_probability: number;
  loser_probability: number;
  prediction_confidence: number;
  model_version: string;
  model_accuracy: string;
  model?: {
    version: string;
    accuracy: string;
    status: string;
  };
  head_to_head: {
    fighter1_wins?: number;
    fighter2_wins?: number;
    last_winner?: string;
    last_method?: string;
  };
  fighter1: {
    name: string;
    record: string;
    image_url: string;
    probability: number;
    win_probability: string;
  };
  fighter2: {
    name: string;
    record: string;
    image_url: string;
    probability: number;
    win_probability: string;
  };
  analysis?: string;
}

export default function FightPredictionsPage() {
  const { toast } = useToast();
  const [fighter1, setFighter1] = useState<FighterStats | null>(null);
  const [fighter2, setFighter2] = useState<FighterStats | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [showPredictionModal, setShowPredictionModal] = useState(false);

  const fetchFighterData = async (fighterName: string): Promise<FighterStats | null> => {
    try {
      if (!fighterName || fighterName === 'undefined') {
        toast({
          title: 'Error',
          description: 'Invalid fighter name',
          variant: 'destructive',
        });
        return null;
      }
      
      console.log(`Fetching fighter data for comparison: ${fighterName}`);
      const response = await fetch(ENDPOINTS.FIGHTER(fighterName));
      if (!response.ok) throw new Error('Fighter not found');
      
      const data = await response.json();
      console.log("Fighter API response:", data);
      
      // Map backend field names to what our frontend expects
      const sanitizedData: FighterStats = {
        name: data?.fighter_name || data?.name || fighterName || '',
        image_url: data?.image_url || '',
        record: data?.Record || data?.record || '',
        height: data?.Height || data?.height || '',
        weight: data?.Weight || data?.weight || '',
        reach: data?.Reach || data?.reach || '',
        stance: data?.STANCE || data?.stance || '',
        slpm: data?.SLpM || data?.SLPM || data?.slpm || '0',
        str_acc: data?.['Str. Acc.'] || data?.str_acc || '0%',
        sapm: data?.SApM || data?.SAPM || data?.sapm || '0',
        str_def: data?.['Str. Def'] || data?.str_def || '0%',
        td_avg: data?.['TD Avg.'] || data?.td_avg || '0',
        td_acc: data?.['TD Acc.'] || data?.td_acc || '0%',
        td_def: data?.['TD Def.'] || data?.td_def || '0%',
        sub_avg: data?.['Sub. Avg.'] || data?.sub_avg || '0',
        ranking: data?.ranking || 0,
      };
      
      console.log("Mapped fighter data for comparison:", sanitizedData);
      
      // Make sure all fields have values
      if (typeof sanitizedData.name !== 'string') sanitizedData.name = fighterName || '';
      if (typeof sanitizedData.image_url !== 'string') sanitizedData.image_url = '';
      if (typeof sanitizedData.record !== 'string') sanitizedData.record = '';
      if (typeof sanitizedData.height !== 'string') sanitizedData.height = '';
      if (typeof sanitizedData.weight !== 'string') sanitizedData.weight = '';
      if (typeof sanitizedData.reach !== 'string') sanitizedData.reach = '';
      if (typeof sanitizedData.stance !== 'string') sanitizedData.stance = '';
      if (typeof sanitizedData.slpm !== 'string') sanitizedData.slpm = '0';
      if (typeof sanitizedData.str_acc !== 'string') sanitizedData.str_acc = '0%';
      if (typeof sanitizedData.sapm !== 'string') sanitizedData.sapm = '0';
      if (typeof sanitizedData.str_def !== 'string') sanitizedData.str_def = '0%';
      if (typeof sanitizedData.td_avg !== 'string') sanitizedData.td_avg = '0';
      if (typeof sanitizedData.td_acc !== 'string') sanitizedData.td_acc = '0%';
      if (typeof sanitizedData.td_def !== 'string') sanitizedData.td_def = '0%';
      if (typeof sanitizedData.sub_avg !== 'string') sanitizedData.sub_avg = '0';
      
      return sanitizedData;
    } catch (error) {
      console.error('Error fetching fighter data:', error);
      toast({
        title: 'Error',
        description: 'Fighter not found. Please try another fighter.',
        variant: 'destructive',
      });
      return null;
    }
  };

  const handleFighter1Select = async (name: string) => {
    if (!name || name === 'undefined') {
      toast({
        title: 'Error',
        description: 'Invalid fighter name selected',
        variant: 'destructive',
      });
      return;
    }
    
    // Clean the fighter name - remove record in parentheses if present
    let cleanName = name;
    if (name.includes('(')) {
      cleanName = name.split('(')[0].trim();
    }
    
    console.log('Fighter 1 selected:', cleanName);
    
    setIsPredicting(true);
    const data = await fetchFighterData(cleanName);
    setFighter1(data);
    setIsPredicting(false);
  };

  const handleFighter2Select = async (name: string) => {
    if (!name || name === 'undefined') {
      toast({
        title: 'Error',
        description: 'Invalid fighter name selected',
        variant: 'destructive',
      });
      return;
    }
    
    // Clean the fighter name - remove record in parentheses if present
    let cleanName = name;
    if (name.includes('(')) {
      cleanName = name.split('(')[0].trim();
    }
    
    console.log('Fighter 2 selected:', cleanName);
    
    setIsPredicting(true);
    const data = await fetchFighterData(cleanName);
    setFighter2(data);
    setIsPredicting(false);
  };

  const getComparisonColor = (val1: number, val2: number) => {
    // Only use yellow if the difference is 0.1 or less
    if (Math.abs(val1 - val2) <= 0.1) return 'text-yellow-500'
    return val1 > val2 ? 'text-green-500' : 'text-red-500'
  }

  const extractNumber = (str: string) => {
    // Safety check for null or undefined
    if (!str) return 0;
    
    // Extract numbers from strings like "5' 7"" or "125 lbs."
    const match = str.match(/(\d+)/);
    return match ? Number(match[1]) : 0;
  }

  const ComparisonRow = ({ label, value1, value2, higherIsBetter = true, unit = '', isPhysicalStat = false }: { 
    label: string;
    value1: any;
    value2: any;
    higherIsBetter?: boolean;
    unit?: string;
    isPhysicalStat?: boolean;
  }) => {
    try {
      // Safety check - ensure values are never undefined before using toString()
      const safeValue1 = value1 !== undefined && value1 !== null ? value1 : '';
      const safeValue2 = value2 !== undefined && value2 !== null ? value2 : '';
      
      // Convert to strings for display
      const strValue1 = String(safeValue1);
      const strValue2 = String(safeValue2);
      
      // Parse numbers for comparison
      let num1 = 0;
      let num2 = 0;
      
      try {
        if (isPhysicalStat) {
          // Extract numbers from strings like "5' 7"" or "125 lbs."
          const match1 = strValue1.match(/(\d+)/);
          const match2 = strValue2.match(/(\d+)/);
          num1 = match1 ? Number(match1[1]) : 0;
          num2 = match2 ? Number(match2[1]) : 0;
        } else if (unit === '%') {
          // Handle percentage values
          num1 = parseFloat(strValue1.replace('%', '')) || 0;
          num2 = parseFloat(strValue2.replace('%', '')) || 0;
        } else {
          // Regular number conversion
          num1 = Number(safeValue1) || 0;
          num2 = Number(safeValue2) || 0;
        }
      } catch (e) {
        console.error('Error parsing comparison values:', e);
        // Default to 0 if parsing fails
        num1 = 0;
        num2 = 0;
      }
      
      // For stats where lower is better, invert the comparison
      const compareVal1 = higherIsBetter ? num1 : -num1;
      const compareVal2 = higherIsBetter ? num2 : -num2;
      
      // Determine color based on comparison
      let color1 = 'text-yellow-500';
      let color2 = 'text-yellow-500';
      
      // Only use different colors if there's a significant difference
      if (Math.abs(num1 - num2) > 0.1) {
        color1 = compareVal1 > compareVal2 ? 'text-green-500' : 'text-red-500';
        color2 = compareVal2 > compareVal1 ? 'text-green-500' : 'text-red-500';
      }
  
      // Remove % from the value if unit is %
      const displayValue1 = unit === '%' ? strValue1.replace('%', '') : strValue1;
      const displayValue2 = unit === '%' ? strValue2.replace('%', '') : strValue2;
  
      return (
        <motion.div 
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-3 gap-2 py-1.5 items-center hover:bg-accent/5 rounded-lg transition-colors"
        >
          <div className={`text-right font-medium ${color1}`}>{displayValue1}</div>
          <div className="text-center text-sm text-muted-foreground font-medium px-2">{label}</div>
          <div className={`text-left font-medium ${color2}`}>{displayValue2}</div>
        </motion.div>
      );
    } catch (error) {
      // Fallback in case of any errors
      console.error('Error rendering comparison row:', error);
      return (
        <motion.div className="grid grid-cols-3 gap-2 py-1.5 items-center">
          <div className="text-right">-</div>
          <div className="text-center text-sm text-muted-foreground">{label}</div>
          <div className="text-left">-</div>
        </motion.div>
      );
    }
  }

  const getPrediction = async (fighter1Name: string, fighter2Name: string) => {
    // Check for valid fighter names
    if (!fighter1Name || !fighter2Name) {
      toast({
        title: 'Error',
        description: 'Please select two fighters to compare',
        variant: 'destructive',
      });
      return;
    }
    
    // Clean names one more time to be extra safe
    let cleanFighter1 = fighter1Name;
    let cleanFighter2 = fighter2Name;
    
    if (cleanFighter1.includes('(')) {
      cleanFighter1 = cleanFighter1.split('(')[0].trim();
    }
    
    if (cleanFighter2.includes('(')) {
      cleanFighter2 = cleanFighter2.split('(')[0].trim();
    }
    
    setIsPredicting(true);
    try {
      console.log(`Making API call with: "${cleanFighter1}" vs "${cleanFighter2}"`);
      
      const response = await fetch(ENDPOINTS.PREDICTION(cleanFighter1, cleanFighter2));

      if (!response.ok) {
        throw new Error('Failed to get prediction');
      }

      const data = await response.json();
      console.log("Prediction API response:", data);
      
      // Validate the prediction data and provide default values
      const validatedPrediction: Prediction = {
        winner: data?.winner || data?.predicted_winner || "Unknown",
        loser: data?.loser || "Unknown",
        winner_probability: data?.winner_probability || data?.confidence || 0,
        loser_probability: data?.loser_probability || 0,
        prediction_confidence: data?.prediction_confidence || data?.confidence || 0,
        model_version: data?.model_version || data?.model?.version || "Unknown",
        model_accuracy: data?.model_accuracy || data?.model?.accuracy || "Unknown",
        head_to_head: {
          fighter1_wins: data?.head_to_head?.fighter1_wins || 0,
          fighter2_wins: data?.head_to_head?.fighter2_wins || 0,
          last_winner: data?.head_to_head?.last_winner || "N/A",
          last_method: data?.head_to_head?.last_method || "N/A",
        },
        fighter1: {
          name: data?.fighter1?.name || cleanFighter1,
          record: data?.fighter1?.record || "0-0-0",
          image_url: data?.fighter1?.image_url || "",
          probability: data?.fighter1?.probability || 0,
          win_probability: data?.fighter1?.win_probability || "0%",
        },
        fighter2: {
          name: data?.fighter2?.name || cleanFighter2,
          record: data?.fighter2?.record || "0-0-0",
          image_url: data?.fighter2?.image_url || "",
          probability: data?.fighter2?.probability || 0,
          win_probability: data?.fighter2?.win_probability || "0%",
        },
        analysis: data?.analysis || "",
      };
      
      // Just use the response as-is - the backend now provides properly formatted win probabilities
      setPrediction(validatedPrediction);
      setShowPredictionModal(true);
    } catch (error) {
      console.error('Error getting prediction:', error);
      toast({
        title: 'Error',
        description: 'Failed to get prediction. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsPredicting(false);
    }
  };

  // Utility function to safely display a fighter name
  const safeDisplayName = (name: any): string => {
    if (!name) return 'Unknown Fighter';
    if (typeof name !== 'string') {
      try {
        return String(name);
      } catch (e) {
        return 'Unknown Fighter';
      }
    }
    return name;
  };

  const PredictionModal = () => (
    <AnimatePresence>
      {showPredictionModal && prediction && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50"
          onClick={() => setShowPredictionModal(false)}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            onClick={(e: React.MouseEvent) => e.stopPropagation()}
            className="bg-card p-8 rounded-xl shadow-2xl w-[800px] mx-4 relative border border-border/50"
          >
            <button
              onClick={() => setShowPredictionModal(false)}
              className="absolute top-4 right-4 text-muted-foreground hover:text-foreground transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
            
            <h3 className="text-2xl font-bold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/70">
              Fight Prediction
            </h3>
            
            {isPredicting ? (
              <div className="flex items-center justify-center py-16">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full"
                />
              </div>
            ) : (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-8"
              >
                {/* Fighter Names and Win Probabilities */}
                <div className="grid grid-cols-[1fr,auto,1fr] gap-8 items-center">
                  <div className="text-center space-y-2">
                    <h4 className="text-xl font-bold">{safeDisplayName(prediction?.fighter1?.name) || 'Fighter 1'}</h4>
                    <div className={`text-lg font-medium ${safeDisplayName(prediction?.fighter1?.name) === safeDisplayName(prediction?.winner) ? 'text-green-500' : 'text-red-500'}`}>
                      {prediction?.fighter1?.win_probability || '0%'}
                    </div>
                    <p className="text-sm text-muted-foreground">{prediction?.fighter1?.record || '0-0-0'}</p>
                  </div>
                  <div className="flex flex-col items-center gap-2">
                    <div className="text-3xl font-bold text-primary">VS</div>
                  </div>
                  <div className="text-center space-y-2">
                    <h4 className="text-xl font-bold">{safeDisplayName(prediction?.fighter2?.name) || 'Fighter 2'}</h4>
                    <div className={`text-lg font-medium ${safeDisplayName(prediction?.fighter2?.name) === safeDisplayName(prediction?.winner) ? 'text-green-500' : 'text-red-500'}`}>
                      {prediction?.fighter2?.win_probability || '0%'}
                    </div>
                    <p className="text-sm text-muted-foreground">{prediction?.fighter2?.record || '0-0-0'}</p>
                  </div>
                </div>

                {/* Winner Banner */}
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-accent/30 to-transparent" />
                  <div className="relative text-center py-4">
                    <h4 className="text-xl font-bold">{safeDisplayName(prediction?.winner) || 'Undecided'}</h4>
                    <p className="text-primary">Predicted Winner</p>
                  </div>
                </div>

                {/* Fight Analysis */}
                <div className="space-y-2">
                  <h5 className="text-lg font-semibold">Fight Analysis</h5>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {prediction?.analysis || "No detailed analysis available for this matchup."}
                  </p>
                </div>

                {/* Head to Head */}
                {prediction?.head_to_head && (prediction?.head_to_head?.fighter1_wins || prediction?.head_to_head?.fighter2_wins) && (
                  <div className="bg-accent/10 rounded-xl p-6">
                    <h4 className="font-semibold text-primary mb-4 text-center">Head to Head History</h4>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div className="text-right font-medium">{prediction?.head_to_head?.fighter1_wins || 0} wins</div>
                      <div className="text-center font-medium text-primary">Previous Fights</div>
                      <div className="text-left font-medium">{prediction?.head_to_head?.fighter2_wins || 0} wins</div>
                    </div>
                    {prediction?.head_to_head?.last_winner && (
                      <div className="text-sm text-muted-foreground text-center mt-4 pt-4 border-t border-border/50">
                        Last fight: <span className="font-medium text-foreground">{safeDisplayName(prediction?.head_to_head?.last_winner)}</span> won by <span className="font-medium text-foreground">{prediction?.head_to_head?.last_method || 'Decision'}</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Model Info */}
                <div className="flex justify-between items-center text-xs text-muted-foreground border-t border-border pt-4">
                  <div>Model Version: {prediction?.model?.version || prediction?.model_version || "1.0"}</div>
                  <div>Model Accuracy: {prediction?.model?.accuracy || prediction?.model_accuracy || "N/A"}</div>
                </div>

                {/* Note */}
                <p className="text-sm text-muted-foreground text-center italic bg-accent/5 rounded-lg p-4">
                  Note: This prediction is based on historical data and statistics. MMA is unpredictable, and any fighter can win on any given night.
                </p>
              </motion.div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );

  // Check fighter data
  const isValidData = (fighter: FighterStats | null): boolean => {
    return fighter !== null && fighter !== undefined && typeof fighter.name === 'string' && fighter.name.trim() !== '';
  }

  // Called when the "Predict Fight Outcome" button is clicked
  const handlePredictClick = () => {
    if (!isValidData(fighter1) || !isValidData(fighter2)) {
      toast({
        title: 'Error',
        description: 'Both fighters must be selected with valid data',
        variant: 'destructive',
      });
      return;
    }
    
    // Safe non-null assertion since we just checked
    let fighter1Name = fighter1!.name;
    let fighter2Name = fighter2!.name;
    
    // Clean names if they have parentheses
    if (fighter1Name.includes('(')) {
      fighter1Name = fighter1Name.split('(')[0].trim();
    }
    
    if (fighter2Name.includes('(')) {
      fighter2Name = fighter2Name.split('(')[0].trim();
    }
    
    console.log(`Predicting fight between cleaned names: "${fighter1Name}" vs "${fighter2Name}"`);
    
    // Call prediction with cleaned names
    getPrediction(fighter1Name, fighter2Name);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-background/80">
      <PredictionModal />
      <div className="container mx-auto px-4 py-4 max-w-5xl animate-in fade-in duration-700">
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-4"
        >
          <h1 className="text-3xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/70">
            Fight Predictions
          </h1>
          <p className="text-muted-foreground">
            Compare fighter statistics and get AI-powered fight predictions.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
          {/* Fighter 1 Selection */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-2"
          >
            <h2 className="text-lg font-semibold text-center mb-1">Fighter 1</h2>
            <div className="animate-in slide-in-from-left duration-700 delay-200">
              <FighterSearch
                onSelectFighter={handleFighter1Select}
                clearSearch={!fighter1}
              />
            </div>
            {fighter1 && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                className="relative aspect-[4/5] rounded-lg overflow-hidden group shadow-lg hover:shadow-xl transition-shadow"
              >
                {fighter1.tap_link ? (
                  <a
                    href={fighter1.tap_link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block w-full h-full"
                  >
                    <img
                      src={fighter1.image_url}
                      alt={fighter1.name}
                      className="w-full h-full object-cover object-top transition-transform duration-500 group-hover:scale-110"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent flex flex-col justify-end p-4 group-hover:from-black/90 transition-all duration-300">
                      <h3 className="text-white text-xl font-bold">{fighter1.name}</h3>
                      <p className="text-white/90">{fighter1.record}</p>
                      {fighter1.ranking && fighter1.ranking !== 99 && (
                        <p className="text-white/90">Rank: #{fighter1.ranking}</p>
                      )}
                    </div>
                  </a>
                ) : (
                  <div className="w-full h-full">
                    <img
                      src={fighter1.image_url}
                      alt={fighter1.name}
                      className="w-full h-full object-cover object-top"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent flex flex-col justify-end p-4 group-hover:from-black/90 transition-all duration-300">
                      <h3 className="text-white text-xl font-bold">{fighter1.name}</h3>
                      <p className="text-white/90">{fighter1.record}</p>
                      {fighter1.ranking && fighter1.ranking !== 99 && (
                        <p className="text-white/90">Rank: #{fighter1.ranking}</p>
                      )}
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </motion.div>

          {/* Stats Comparison */}
          <Card className="relative h-fit shadow-lg hover:shadow-xl transition-shadow">
            <CardContent className="p-3">
              {fighter1 && fighter2 ? (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                  className="space-y-3"
                >
                  <motion.div 
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 200, damping: 15 }}
                    className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-background rounded-full p-2 shadow-lg"
                  >
                    <Swords className="w-5 h-5 text-primary" />
                  </motion.div>
                  
                  <h3 className="text-base font-semibold text-center mb-2 mt-3">Stats Comparison</h3>
                  
                  <div className="space-y-3">
                    {/* Physical Stats */}
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.1 }}
                    >
                      <h4 className="text-sm font-medium text-primary text-center mb-1">Physical Stats</h4>
                      <div className="space-y-0.5">
                        <ComparisonRow
                          label="Height"
                          value1={fighter1.height}
                          value2={fighter2.height}
                          isPhysicalStat
                        />
                        <ComparisonRow
                          label="Weight"
                          value1={fighter1.weight}
                          value2={fighter2.weight}
                          isPhysicalStat
                        />
                        <ComparisonRow
                          label="Reach"
                          value1={fighter1.reach}
                          value2={fighter2.reach}
                          isPhysicalStat
                        />
                      </div>
                    </motion.div>

                    {/* Striking Stats */}
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.2 }}
                    >
                      <h4 className="text-sm font-medium text-primary text-center mb-1">Striking</h4>
                      <div className="space-y-0.5">
                        <ComparisonRow
                          label="Strikes Landed/min"
                          value1={fighter1.slpm}
                          value2={fighter2.slpm}
                        />
                        <ComparisonRow
                          label="Strike Accuracy"
                          value1={fighter1.str_acc}
                          value2={fighter2.str_acc}
                          unit="%"
                        />
                        <ComparisonRow
                          label="Strikes Absorbed/min"
                          value1={fighter1.sapm}
                          value2={fighter2.sapm}
                          higherIsBetter={false}
                        />
                        <ComparisonRow
                          label="Strike Defense"
                          value1={fighter1.str_def}
                          value2={fighter2.str_def}
                          unit="%"
                        />
                      </div>
                    </motion.div>
                    
                    {/* Grappling Stats */}
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                    >
                      <h4 className="text-sm font-medium text-primary text-center mb-1">Grappling</h4>
                      <div className="space-y-0.5">
                        <ComparisonRow
                          label="Takedowns"
                          value1={fighter1.td_avg}
                          value2={fighter2.td_avg}
                        />
                        <ComparisonRow
                          label="Takedown Accuracy"
                          value1={fighter1.td_acc}
                          value2={fighter2.td_acc}
                          unit="%"
                        />
                        <ComparisonRow
                          label="Takedown Defense"
                          value1={fighter1.td_def}
                          value2={fighter2.td_def}
                          unit="%"
                        />
                        <ComparisonRow
                          label="Sub. Average"
                          value1={fighter1.sub_avg}
                          value2={fighter2.sub_avg}
                        />
                      </div>
                    </motion.div>

                    {/* Predict Button */}
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.4 }}
                      className="pt-2 flex justify-center"
                    >
                      <Button
                        onClick={handlePredictClick}
                        className="w-full"
                      >
                        Predict Fight Outcome
                      </Button>
                    </motion.div>
                  </div>
                </motion.div>
              ) : (
                <div className="h-[350px] flex flex-col items-center justify-center text-muted-foreground">
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.3 }}
                    className="text-center"
                  >
                    <motion.div
                      animate={{ 
                        rotate: [0, -10, 10, -10, 0],
                        scale: [1, 1.1, 1]
                      }}
                      transition={{ 
                        duration: 2,
                        repeat: Infinity,
                        repeatDelay: 1
                      }}
                    >
                      <ArrowLeftRight className="w-6 h-6 mb-2 mx-auto text-primary" />
                    </motion.div>
                    <p className="text-sm">Select two fighters to compare stats</p>
                  </motion.div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Fighter 2 Selection */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-2"
          >
            <h2 className="text-lg font-semibold text-center mb-1">Fighter 2</h2>
            <motion.div
              key={fighter1 ? 'search-active' : 'search-inactive'}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              className="animate-in slide-in-from-right duration-700 delay-200"
            >
              <FighterSearch
                onSelectFighter={handleFighter2Select}
                clearSearch={!fighter2}
              />
            </motion.div>
            {fighter2 && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                className="relative aspect-[4/5] rounded-lg overflow-hidden group shadow-lg hover:shadow-xl transition-shadow"
              >
                {fighter2.tap_link ? (
                  <a
                    href={fighter2.tap_link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block w-full h-full"
                  >
                    <img
                      src={fighter2.image_url}
                      alt={fighter2.name}
                      className="w-full h-full object-cover object-top transition-transform duration-500 group-hover:scale-110"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent flex flex-col justify-end p-4 group-hover:from-black/90 transition-all duration-300">
                      <h3 className="text-white text-xl font-bold">{fighter2.name}</h3>
                      <p className="text-white/90">{fighter2.record}</p>
                      {fighter2.ranking && fighter2.ranking !== 99 && (
                        <p className="text-white/90">Rank: #{fighter2.ranking}</p>
                      )}
                    </div>
                  </a>
                ) : (
                  <div className="w-full h-full">
                    <img
                      src={fighter2.image_url}
                      alt={fighter2.name}
                      className="w-full h-full object-cover object-top"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent flex flex-col justify-end p-4 group-hover:from-black/90 transition-all duration-300">
                      <h3 className="text-white text-xl font-bold">{fighter2.name}</h3>
                      <p className="text-white/90">{fighter2.record}</p>
                      {fighter2.ranking && fighter2.ranking !== 99 && (
                        <p className="text-white/90">Rank: #{fighter2.ranking}</p>
                      )}
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  )
} 