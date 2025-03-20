"use client"

import { useState } from "react"
import { FighterSearch } from "@/components/fighter-search"
import { FighterDetails } from "@/components/fighter-details"

export default function FightersPage() {
  const [selectedFighter, setSelectedFighter] = useState<string | null>(null)

  return (
    <div className="min-h-screen">
      <div className="container mx-auto px-4 py-12 max-w-5xl flex flex-col items-center animate-in fade-in duration-700">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">MMA Fighters</h1>
          <p className="text-lg text-muted-foreground">
            Browse and search through fighter statistics and rankings.
          </p>
        </div>
        <div className="w-full max-w-md mb-12 animate-in slide-in-from-top duration-700 delay-200">
          <FighterSearch 
            onSelectFighter={(fighter) => {
              setSelectedFighter(fighter)
            }}
            clearSearch={!selectedFighter} 
          />
        </div>

        {selectedFighter && (
          <div className="w-full">
            <FighterDetails fighterName={selectedFighter} />
          </div>
        )}
      </div>
    </div>
  )
} 