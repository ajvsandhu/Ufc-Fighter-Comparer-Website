"use client"

import * as React from "react"
import { Check, Search } from "lucide-react"
import { cn } from "@/lib/utils"
import { Input } from "@/components/ui/input"
import { ENDPOINTS } from "@/lib/api-config"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandItem,
} from "@/components/ui/command"

interface FighterSearchProps {
  onSelectFighter: (fighter: string) => void
  clearSearch?: boolean
}

export function FighterSearch({ onSelectFighter, clearSearch }: FighterSearchProps) {
  const [searchTerm, setSearchTerm] = React.useState("")
  const [fighters, setFighters] = React.useState<string[]>([])
  const [isLoading, setIsLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [showSuggestions, setShowSuggestions] = React.useState(false)
  const wrapperRef = React.useRef<HTMLDivElement>(null)

  // Handle click outside to close suggestions
  React.useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setShowSuggestions(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  // Clear search when clearSearch prop changes
  React.useEffect(() => {
    if (clearSearch) {
      setSearchTerm("")
      setFighters([])
      setShowSuggestions(false)
    }
  }, [clearSearch])

  // Fetch fighters when search term changes
  React.useEffect(() => {
    if (!searchTerm.trim()) {
      setFighters([]);
      setShowSuggestions(false);
      return;
    }

    const fetchFighters = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch(ENDPOINTS.FIGHTERS_SEARCH(searchTerm.trim()), {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
          }
        });
        
        if (!response.ok) {
          throw new Error('Failed to fetch fighters');
        }
        
        const data = await response.json();
        
        // Add extra safety checks for the fighters data
        let fightersList: string[] = [];
        if (data && data.fighters && Array.isArray(data.fighters)) {
          // Filter out any null or undefined values and ensure all items are strings
          fightersList = data.fighters
            .filter((fighter: any) => fighter != null)
            .map((fighter: any) => String(fighter));
        }
        
        // Limit to 5 suggestions
        setFighters(fightersList.slice(0, 5));
        setShowSuggestions(true);
      } catch (err) {
        console.error('Search error:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch fighters');
        setFighters([]);
      } finally {
        setIsLoading(false);
      }
    }

    const debounceTimer = setTimeout(fetchFighters, 300);
    return () => clearTimeout(debounceTimer);
  }, [searchTerm]);

  const formatFighterDisplay = (fighter: string) => {
    // Add safety check for fighter being undefined or null
    if (!fighter) return '';
    
    if (!fighter.includes('(')) return fighter;

    try {
      const [baseName, ...rest] = fighter.split('(');
      const info = '(' + rest.join('(');
      return (
        <div className="flex flex-col">
          <span className="font-medium">{baseName.trim()}</span>
          <span className="text-sm text-muted-foreground">{info}</span>
        </div>
      );
    } catch (err) {
      // If any error occurs, return the fighter name as-is
      console.error('Error formatting fighter name:', err);
      return fighter;
    }
  }

  return (
    <div ref={wrapperRef} className="relative w-full">
      <div className="relative">
        <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search fighters..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-9 pr-4"
          onFocus={() => setShowSuggestions(true)}
        />
      </div>
      {showSuggestions && (searchTerm.trim() || isLoading || error) && (
        <div className="absolute top-full w-full z-50 mt-1 rounded-md border bg-popover text-popover-foreground shadow-md">
          <Command className="rounded-md" shouldFilter={false}>
            {error && (
              <p className="p-2 text-sm text-destructive">{error}</p>
            )}
            {isLoading ? (
              <p className="p-2 text-sm text-muted-foreground">Loading...</p>
            ) : fighters.length === 0 ? (
              <CommandEmpty>No fighters found.</CommandEmpty>
            ) : (
              <CommandGroup>
                {fighters.map((fighter) => (
                  <CommandItem
                    key={fighter}
                    value={fighter}
                    onSelect={(currentValue: string) => {
                      onSelectFighter(currentValue)
                      setShowSuggestions(false)
                      setSearchTerm("") // Clear search term after selection
                    }}
                    className="cursor-pointer"
                  >
                    <Check
                      className={cn(
                        "mr-2 h-4 w-4",
                        searchTerm === fighter ? "opacity-100" : "opacity-0"
                      )}
                    />
                    {formatFighterDisplay(fighter)}
                  </CommandItem>
                ))}
              </CommandGroup>
            )}
          </Command>
        </div>
      )}
    </div>
  )
} 